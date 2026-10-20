"""Export the data behind the page's interactive visuals to static/data/.

* ``g1_mesh.json``: the G1's visual meshes, decimated, one piece per geom expressed in its body's
  frame, with its colour (from the scene's materials). Little-endian float32 positions, uint32 indices.

The binary parts are stored base64 in each file's ``data`` field.
* ``denoise.json``: one plan of a recorded rollout (``sub7_045_final.npz`` env 0, the paper's Student + SD:DPO +
  DA:PPO planner) at ten noise levels. Each level is the forward-noised plan
  ``sqrt(abar_k) z0 + sqrt(1 - abar_k) eps`` with one fixed ``eps`` in per-feature normalised coordinates, which is
  also the path an ideal deterministic (DDIM) denoiser takes from that noise back to the plan.
* ``sbto_paths.json``: the robot's x-y path for every SBTO solution of the 29 pick-carry-place motions (the paper's
  augmentation figure), in centimetres, with the lowest-cost solution of each motion.

Positions are relative to the rollout's env origin, MuJoCo axes (z up).

    cd ../zerosho && uv run --no-sync --with fast-simplification python ../diplo-website/tools/export_web_data.py
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "data"
ZEROSHO = Path("/home/breakout/workspace/zerosho")
REC = ZEROSHO / "outputs/video/rollouts/sub7_045_final.npz"
ENV = 0
SBTO = Path("/home/breakout/workspace/sbto/test_data/pick_place_diffusion_done_combined")
KEEP = 0.14          # fraction of mesh faces kept
STRIDE = 2           # export every 2nd control frame (25 fps)
DENOISE_CHUNK = 3    # mid-carry, the plan of the video's plan shot
LEVELS = 10


def dump(name: str, meta: dict, *arrays: np.ndarray) -> None:
    """``name``.json: ``meta`` plus the arrays' little-endian bytes, concatenated, as base64 in ``data`` (static hosts
    and the Artifact preview serve .json but not raw .bin)."""
    raw = b"".join(a.tobytes() for a in arrays)
    json.dump({**meta, "data": base64.b64encode(raw).decode()}, open(OUT / f"{name}.json", "w"), separators=(",", ":"))


def body_ids(m: mujoco.MjModel) -> list[int]:
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
    robot = [b for b, n in enumerate(names) if n.startswith("robot/")]
    return robot + [names.index("box/root")]


def export_meshes(m: mujoco.MjModel, bodies: list[int]) -> None:
    """One piece per visual mesh geom: its body's index in ``bodies``, its vertex/index ranges and its colour."""
    pos_chunks, idx_chunks, meta = [], [], []
    vofs = iofs = 0
    for bi, b in enumerate(bodies[:-1]):  # the object is drawn by the page
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - KEEP)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[g])
            v = (v @ rot.reshape(3, 3).T + m.geom_pos[g]).astype(np.float32)
            f = f.astype(np.uint32).ravel()
            mat = m.geom_matid[g]
            rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
            pos_chunks.append(v.ravel())
            idx_chunks.append(f)
            meta.append({"body": bi, "v": [vofs, len(v)], "i": [iofs, len(f)],
                         "rgb": [round(float(x), 3) for x in rgba[:3]]})
            vofs += len(v)
            iofs += len(f)
    pos = np.concatenate(pos_chunks).astype("<f4")
    idx = np.concatenate(idx_chunks).astype("<u4")
    dump("g1_mesh", {"pieces": meta, "positions_bytes": int(pos.nbytes), "vertices": vofs, "indices": iofs}, pos, idx)
    print(f"mesh: {len(meta)} pieces, {vofs} vertices, {iofs // 3} triangles, {(pos.nbytes + idx.nbytes) / 1e6:.2f} MB")


def fk(m: mujoco.MjModel, d: mujoco.MjData, qpos: np.ndarray, bodies: list[int], origin: np.ndarray) -> np.ndarray:
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    out = np.concatenate([d.xpos[bodies] - origin, d.xquat[bodies]], axis=1)
    return out.astype(np.float32)  # (bodies, 7)


def plan_qpos(z, c: int, j: int, origin: np.ndarray, nq: int, override=None, env: int = ENV) -> np.ndarray:
    """qpos of plan frame ``j`` of chunk ``c`` (as chunk_buffer.qpos_from_chunk_frame), world frame."""
    root_p = z["plan_body_pos"][c, env, j, 0]
    root_q = z["plan_body_quat"][c, env, j, 0]
    joints = z["plan_joint_pos"][c, env, j]
    obj_p = z["plan_object_pos_w"][c, env, j]
    obj_q = z["plan_object_quat_w"][c, env, j]
    if override is not None:
        root_p, root_q, joints, obj_p, obj_q = override
    q = np.zeros(nq)
    q[0:3], q[3:7], q[7:36] = root_p + origin, root_q, joints
    q[36:39], q[39:43] = obj_p + origin, obj_q
    return q


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., :1])
    return np.where(s > 1e-9, q[..., 1:] / np.maximum(s, 1e-9) * ang, 2 * q[..., 1:])


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    ang = np.linalg.norm(r, axis=-1, keepdims=True)
    axis = np.where(ang > 1e-9, r / np.maximum(ang, 1e-9), 0)
    return np.concatenate([np.cos(ang / 2), axis * np.sin(ang / 2)], axis=-1)


def export_replay(m: mujoco.MjModel, bodies: list[int], z, env: int, name: str, end: int | None = None) -> list[int]:
    """``name``.json: env ``env`` of the recorded rollout ``z`` up to control frame ``end`` (all of it by default),
    every second control frame executed and every plan the planner generated, as body poses. Returns the plans'
    first control frames."""
    meta = json.loads(str(z["meta"]))
    d = mujoco.MjData(m)
    origin = z["env_origins"][env].astype(np.float64)
    nq = m.nq
    T = z["qpos"].shape[0] if end is None else end + 1
    frames = list(range(0, T, STRIDE))
    executed = np.stack([fk(m, d, z["qpos"][t, env].astype(np.float64), bodies, origin) for t in frames])
    chunk, cursor = z["chunk"][:T], z["cursor"][:T]
    n_chunks = z["plan_joint_pos"].shape[0] if end is None else int(chunk.max()) + 1
    n_plan = z["plan_joint_pos"].shape[2]
    starts = [int(np.argmax((chunk == c) & (cursor == 0))) for c in range(n_chunks)]
    plans = np.stack([np.stack([fk(m, d, plan_qpos(z, c, j, origin, nq, env=env), bodies, origin)
                                for j in range(0, n_plan, STRIDE)]) for c in range(n_chunks)])
    goal = np.concatenate([z["goal_pos"][0, env] - origin, z["goal_quat"][0, env]]).astype(np.float32)
    motion = meta["motion_names"][int(z["motion_id"][0, env])] if "motion_id" in z.files else meta["motion_names"][0]
    dump(name, {
        "fps": 1 / (meta["control_dt"] * STRIDE), "frames": len(frames), "bodies": len(bodies),
        "plan_chunks": n_chunks, "plan_frames": plans.shape[1], "plan_stride": STRIDE, "act_steps": meta["act_steps"],
        "chunk_start": [s // STRIDE for s in starts], "chunk_start_control": starts,
        "chunk": chunk[frames].tolist(), "cursor": (cursor[frames] // STRIDE).tolist(),
        "goal": goal.round(4).tolist(), "box_half": [0.155, 0.155, 0.17],
        "motion": motion, "checkpoint": meta["generator_checkpoint"],
    }, executed.astype("<f4"), plans.astype("<f4"))
    print(f"{name}: {motion}, {len(frames)} frames, {n_chunks} plans x {plans.shape[1]} frames, starts {starts}")
    return starts


def export_replay_and_denoise(m: mujoco.MjModel, bodies: list[int]) -> None:
    z = np.load(REC, allow_pickle=True)
    d = mujoco.MjData(m)
    origin = z["env_origins"][ENV].astype(np.float64)
    nq = m.nq
    n_plan = z["plan_joint_pos"].shape[2]
    starts = [int(np.argmax((z["chunk"] == c) & (z["cursor"] == 0))) for c in range(z["plan_joint_pos"].shape[0])]
    goal = np.concatenate([z["goal_pos"][0, ENV] - origin, z["goal_quat"][0, ENV]]).astype(np.float32)

    # Ten noise levels of one plan (cosine schedule), per-feature normalised about the plan's own mean.
    c = DENOISE_CHUNK
    rp = z["plan_body_pos"][c, ENV, :, 0].astype(np.float64)
    rr = quat_to_rotvec(z["plan_body_quat"][c, ENV, :, 0].astype(np.float64))
    jp = z["plan_joint_pos"][c, ENV].astype(np.float64)
    op = z["plan_object_pos_w"][c, ENV].astype(np.float64)
    orr = quat_to_rotvec(z["plan_object_quat_w"][c, ENV].astype(np.float64))
    x0 = np.concatenate([rp, rr, jp, op, orr], axis=1)                  # (64, 41)
    scale = np.concatenate([[0.12] * 3, [0.35] * 3, [0.55] * 29, [0.12] * 3, [0.5] * 3])
    mu = x0.mean(0)
    z0 = (x0 - mu) / scale
    eps = np.random.default_rng(7).standard_normal(z0.shape)
    s = 0.008
    t = np.arange(LEVELS + 1) / LEVELS
    abar = (np.cos((t + s) / (1 + s) * np.pi / 2) / np.cos(s / (1 + s) * np.pi / 2)) ** 2
    abar[-1] = 0.0
    show = list(range(0, n_plan, 8)) + [n_plan - 1]                       # the frames drawn as holograms
    levels = []
    for k in range(LEVELS + 1):
        x = mu + scale * (np.sqrt(abar[k]) * z0 + np.sqrt(1 - abar[k]) * eps)
        poses = []
        for j in show:
            f = x[j]
            ov = (f[0:3], rotvec_to_quat(f[3:6]), f[6:35], f[35:38], rotvec_to_quat(f[38:41]))
            poses.append(fk(m, d, plan_qpos(z, c, j, origin, nq, ov), bodies, origin))
        levels.append(np.stack(poses))
    levels = np.stack(levels)                                             # (11, frames, bodies, 7)
    t0 = starts[c]
    now = fk(m, d, z["qpos"][t0, ENV].astype(np.float64), bodies, origin)
    history = np.stack([fk(m, d, z["qpos"][max(0, t0 - h), ENV].astype(np.float64), bodies, origin) for h in (24, 12)])
    dump("denoise", {"levels": LEVELS + 1, "frames": len(show), "frame_index": show, "bodies": len(bodies),
                     "abar": abar.round(5).tolist(), "goal": goal.round(4).tolist(), "box_half": [0.155, 0.155, 0.17],
                     "history": 2}, levels.astype("<f4"), now.astype("<f4"), history.astype("<f4"))
    print(f"denoise: plan {c} (starts at control frame {t0}), {LEVELS + 1} levels x {len(show)} frames")


def export_sbto_paths() -> None:
    demos = []
    total = 0
    for name in sorted(os.listdir(SBTO)):
        if "__" in name:
            continue
        p = SBTO / name / "best_trajectory_rand.npz"
        if not p.is_file():
            continue
        with np.load(p) as z:
            xy = z["root_pos"][..., :2]
            best = int(np.argmin(z["cost"]))
        n = xy.shape[1]
        pick = np.unique(np.linspace(0, n - 1, 48).round().astype(int))
        paths = (xy[:, pick] * 100).round().astype(int)
        demos.append({"name": name.replace("_original", ""), "best": best, "paths": paths.reshape(len(xy), -1).tolist()})
        total += len(xy)
    json.dump({"demos": demos, "solutions": total}, open(OUT / "sbto_paths.json", "w"), separators=(",", ":"))
    print(f"sbto: {len(demos)} motions, {total} solutions")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(REC, allow_pickle=True)
    scene = REC.parent / json.loads(str(z["meta"]))["scene"]
    m = mujoco.MjModel.from_binary_path(str(scene))
    bodies = body_ids(m)
    export_meshes(m, bodies)
    export_replay_and_denoise(m, bodies)
    export_sbto_paths()
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:18s} {f.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()
