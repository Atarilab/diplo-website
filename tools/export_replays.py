"""The rollouts of the page's 3D replay: the paper's final planner (Student + SD:DPO + DA:PPO, pick-carry-place) on
six motions, from zerosho's ``outputs/video/rollouts/pp29x8_final.npz`` (all 29 motions, eight rollouts each) and
``pp29_final.npz`` (two each). Each rollout is cut 1.5 s after the object comes to rest; the first is written as
``replay.json`` (the one the page shows first), the others as ``replay_<n>.json``, and ``replays.json`` lists them all
with a label from the motion's carry distance and the object's turn.

    cd ../zerosho && uv run --no-sync --with fast-simplification python ../diplo-website/tools/export_replays.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_web_data import OUT, REC, body_ids, export_replay  # noqa: E402

# (recording, env): the least staggering successful rollout of six walking carries, from 0.5 m to 1.6 m and 38° to
# 157°, ranked over both recordings by short shuffling foot moves, foot travel without lifting the foot, torso roll,
# pelvis sway, foot speed, leg jerk and box spin (tools/stagger_scores.py; the turn-in-place motions all pivot on
# their feet and barely lift the box, so none is shown)
PICKS = (
    ("pp29x8_final.npz", 152),  # sub10_083, 1.5 m, 157° turn
    ("pp29_final.npz", 45),     # sub3_003, 1.4 m, 55°
    ("pp29x8_final.npz", 105),  # sub3_005, 1.6 m, 38°
    ("pp29_final.npz", 6),      # sub10_082, 1.4 m, 128°
    ("pp29x8_final.npz", 50),   # sub3_028, 1.2 m, 106°
    ("pp29_final.npz", 48),     # sub3_016, 0.5 m, 47°
)
SETTLE = 75  # control frames shown after the object comes to rest


def yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def describe(z, env: int, end: int) -> dict:
    """The motion's carry distance and the object's turn, from where it starts to the goal."""
    q = z["qpos"][: end + 1, env]
    carry = float(np.linalg.norm(z["goal_pos"][0, env, :2] - q[0, 36:38]))
    turn = abs((yaw(z["goal_quat"][0, env]) - yaw(q[0, 39:43]) + 180) % 360 - 180)
    label = f"Turn in place, {turn:.0f}°" if carry < 0.25 else f"{carry:.1f} m, {turn:.0f}° turn"
    return {"carry_m": round(carry, 2), "turn_deg": round(turn), "label": label}


def settled_end(z, env: int) -> int:
    """1.5 s after the object's last move (more than 1 mm per control frame)."""
    op = z["qpos"][:, env, 36:39]
    moving = np.flatnonzero(np.linalg.norm(np.diff(op, axis=0), axis=1) > 1e-3)
    return int(min(len(op) - 1, (moving[-1] + 1 if len(moving) else 0) + SETTLE))


class Recording(dict):
    """A recording read into memory once (``np.load`` decompresses an array on every access)."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=True) as f:
            super().__init__(f)
        self.files = list(self)


def main() -> None:
    index, model, recs = [], {}, {}
    for n, (rec, env) in enumerate(PICKS):
        if rec not in recs:
            recs[rec] = Recording(REC.parent / rec)
        z = recs[rec]
        meta = json.loads(str(z["meta"]))
        if meta["scene"] not in model:
            m = mujoco.MjModel.from_binary_path(str(REC.parent / meta["scene"]))
            model[meta["scene"]] = (m, body_ids(m))
        m, bodies = model[meta["scene"]]
        name = "replay" if n == 0 else f"replay_{n}"
        end = settled_end(z, env)
        export_replay(m, bodies, z, env, name, end)
        index.append({"file": f"{name}.json", **describe(z, env, end), "motion": meta["motion_names"][int(z["motion_id"][0, env])]})
    json.dump({"rollouts": index}, open(OUT / "replays.json", "w"), indent=1)
    for r in index:
        print(f"  {r['file']:16s} {r['motion']:30s} {r['label']}")


if __name__ == "__main__":
    main()
