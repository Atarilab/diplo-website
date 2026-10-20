"""Staggering scores for recorded rollouts (zerosho's record_rollout.py .npz), behind export_replays.py's PICKS: per env,
up to 1.5 s after the box comes to rest. Lower is smoother; the picks are the successful rollouts lowest on the z-scored
mean of short_per_m (short / (1 + carry_m)), shuffle, roll, sway, peak, jerk, -lift and box_w / 2.

steps     swing phases of either foot (foot > 3 cm above its stance height for > 0.1 s)
steps_per extra steps: steps / (2 + carry_m / 0.25 + turn_deg / 30)   (a rough "needed" count)
slide     horizontal travel of the feet while in stance, m
wobble    RMS roll/pitch angular velocity of the torso, rad/s
jerk      RMS second difference of the leg joints, rad/s^2
sway      RMS lateral acceleration of the pelvis (in its heading frame), m/s^2
box_w     RMS angular velocity of the box while lifted, rad/s

    cd ../zerosho && uv run --no-sync python ../diplo-website/tools/stagger_scores.py outputs/video/rollouts/pp29x8_final.npz
    (writes <npz stem>_stagger.json to the current directory)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

DT = 0.02
SETTLE = 75


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_angvel(q):
    """body-frame angular velocity from consecutive wxyz quats (T, 4) -> (T-1, 3)"""
    q0, q1 = q[:-1], q[1:]
    # dq = conj(q0) * q1
    w0, v0 = q0[:, :1], -q0[:, 1:]
    w1, v1 = q1[:, :1], q1[:, 1:]
    w = w0 * w1 - np.sum(v0 * v1, axis=1, keepdims=True)
    v = w0 * v1 + w1 * v0 + np.cross(v0, v1)
    s = np.sign(w)
    s[s == 0] = 1
    w, v = w * s, v * s
    ang = 2 * np.arctan2(np.linalg.norm(v, axis=1, keepdims=True), w)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9) * ang, 2 * v) / DT


def settled_end(qpos):
    op = qpos[:, 36:39]
    moving = np.flatnonzero(np.linalg.norm(np.diff(op, axis=0), axis=1) > 1e-3)
    return int(min(len(op) - 1, (moving[-1] + 1 if len(moving) else 0) + SETTLE))


def swings(h, thresh=0.03, min_len=5):
    up = h > thresh
    d = np.diff(np.r_[0, up.astype(int), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (e - s) >= min_len
    return s[keep], e[keep], up


def score(m, z, env):
    d = mujoco.MjData(m)
    body = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)}
    feet = [body["robot/left_ankle_roll_link"], body["robot/right_ankle_roll_link"]]
    torso, pelvis = body["robot/torso_link"], body["robot/pelvis"]
    qall = z["qpos"][:, env].astype(np.float64)
    ep = z["episode_length"][:, env]
    end = settled_end(qall)
    reset = bool(np.any(np.diff(ep[: end + 1]) < 0))
    q = qall[: end + 1]
    T = len(q)
    fp = np.zeros((T, 2, 3))
    tq = np.zeros((T, 4))
    for t in range(T):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        fp[t] = d.xpos[feet]
        tq[t] = d.xquat[torso]
    out = {}
    nsteps, slide = 0, 0.0
    n_moves = n_short = 0
    travel = shuffle = 0.0
    peaks, lifts, strides = [], [], []
    for f in range(2):
        h = fp[:, f, 2] - np.percentile(fp[:, f, 2], 5)
        s, e, up = swings(h)
        nsteps += len(s)
        stance = ~up[1:] & ~up[:-1]
        step = np.linalg.norm(np.diff(fp[:, f, :2], axis=0), axis=1)
        slide += float(np.sum(step[stance]))
        travel += float(step.sum())
        shuffle += float(step[(h[1:] < 0.015) & (h[:-1] < 0.015)].sum())
        # foot moves: horizontal speed > 0.2 m/s, gaps under 3 frames merged
        mv = step / DT > 0.2
        idx = np.flatnonzero(mv)
        if len(idx):
            groups = np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1)
            for g in groups:
                a, b = g[0], g[-1] + 1
                disp = float(np.linalg.norm(fp[b, f, :2] - fp[a, f, :2]))
                if disp < 0.02:
                    continue
                n_moves += 1
                n_short += disp < 0.08
                peaks.append(float(step[a:b].max() / DT))
                lifts.append(float(h[a:b + 1].max()))
                strides.append(disp)
    carry = float(np.linalg.norm(z["goal_pos"][0, env, :2] - q[0, 36:38]))
    turn = float(abs((np.degrees(yaw(z["goal_quat"][0, env]) - yaw(q[0, 39:43])) + 180) % 360 - 180))
    need = 2 + carry / 0.25 + turn / 30
    w = quat_angvel(tq)
    roll = float(np.sqrt(np.mean(w[:, 0] ** 2)))  # torso body x: forward axis on the G1, so this is the side-to-side rate
    pv = np.diff(q[:, :3], axis=0) / DT
    pa = np.diff(pv, axis=0) / DT
    hd = yaw(q[1:-1, 3:7])
    lat = -np.sin(hd) * pa[:, 0] + np.cos(hd) * pa[:, 1]
    legs = q[:, 7:19]
    jerk = np.diff(legs, n=2, axis=0) / DT**2
    lifted = q[:, 38] > q[0, 38] + 0.05
    bw = quat_angvel(q[:, 39:43])
    bw = bw[lifted[1:]] if lifted[1:].any() else bw[:1] * 0
    # success: box within 15 cm and 30 deg of the goal at the end of the window
    pos_err = float(np.linalg.norm(q[-1, 36:39] - z["goal_pos"][0, env]))
    gq, bq = z["goal_quat"][0, env].astype(np.float64), q[-1, 39:43]
    ang_err = float(np.degrees(2 * np.arccos(min(1.0, abs(np.dot(gq, bq) / np.linalg.norm(bq))))))
    fell = bool(np.min(q[:, 2]) < 0.5)
    out.update(
        env=int(env), motion_id=int(z["motion_id"][0, env]), end=end, dur_s=round(end * DT, 2),
        carry_m=round(carry, 2), turn_deg=round(turn), steps=nsteps, steps_per=round(nsteps / need, 2),
        slide=round(slide, 3), moves=n_moves, short=int(n_short), shuffle=round(shuffle / max(travel, 1e-6), 3),
        travel=round(travel, 2), stride=round(float(np.mean(strides)) if strides else 0.0, 3),
        lift=round(float(np.mean(lifts)) if lifts else 0.0, 3), peak=round(float(np.mean(peaks)) if peaks else 0.0, 2),
        roll=round(roll, 3), wobble=round(float(np.sqrt(np.mean(w[:, :2] ** 2))), 3),
        jerk=round(float(np.sqrt(np.mean(jerk**2))), 1), sway=round(float(np.sqrt(np.mean(lat**2))), 2),
        box_w=round(float(np.sqrt(np.mean(bw**2))) if len(bw) else 0.0, 3),
        pos_err=round(pos_err, 3), ang_err=round(ang_err, 1), reset=reset, fell=fell,
        ok=bool(pos_err < 0.15 and ang_err < 30 and not reset and not fell),
    )
    return out


def main():
    here = Path.cwd()
    for p in sys.argv[1:]:
        p = Path(p)
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        names = meta["motion_names"]
        if isinstance(names, str):
            names = eval(names)
        m = mujoco.MjModel.from_binary_path(str(p.parent / meta["scene"]))
        rows = []
        for env in range(z["qpos"].shape[1]):
            r = score(m, z, env)
            r["motion"] = names[r["motion_id"]]
            rows.append(r)
        json.dump(rows, open(here / f"{p.stem}_stagger.json", "w"), indent=1)
        print(f"{p.name}: {len(rows)} envs, {sum(r['ok'] for r in rows)} ok")


if __name__ == "__main__":
    main()
