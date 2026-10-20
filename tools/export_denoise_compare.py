"""Export the reverse denoising chains of the paper's two pretrained planners for the page's SO(3) vs Euclidean
comparison (paper Table II, tab:so3-euclid-levels): ``static/data/denoise_cmp.json``.

The two planners differ only in how they diffuse the root and object orientations: on SO(3)
(``ppaug_tube62_sym8``, the paper's pretrained planner) or as flat 6D coordinates re-orthonormalised at decode
(``ppaug_tube62_euclid``, same recipe with ``model.diffusion.so3.enable=false``). Both get the same conditioning,
built as pretraining builds it (zerosho ``scripts/eval_generator_rotation_fidelity.py``): the robot's state history
from the motion's SBTO solution, the IK-retargeted human motion ahead, the final object pose as the goal, for one
chunk of ``sub7_largebox_045``. Each is sampled from the same seeds with its baked solver (10 steps), and every
state of its reverse chain (k = 10, the prior, down to k = 0, the plan) is decoded to robot and object poses and
posed by forward kinematics, as ``denoise.json`` stores them: the first seed in ``denoise_cmp.json`` (with the
shared state, goal and the number of samples), seed ``i`` in ``denoise_cmp_<i>.json`` (the page's "New noise").

    cd ../zerosho && uv run --no-sync --with fast-simplification python ../diplo-website/tools/export_denoise_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_web_data import REC, body_ids, dump, fk  # noqa: E402

from zerosho.data.dataset import TrajectoryPairDataset  # noqa: E402
from zerosho.generators.motion_repr import MotionEncoder, build_sliding_window_condition, required_raw_fields  # noqa: E402
from zerosho.generators.sliding_window import load_generator_checkpoint  # noqa: E402

ARMS = {
    "so3": "logs/generator/diffusion/20260825_191150_ppaug_tube62_sym8/step_0150000.pt",
    "euclid": "logs/generator/diffusion/20260828_183723_ppaug_tube62_euclid/step_0150000.pt",
}
DATA = "data/pick-place-tracked-original"
MOTION = "sub7_largebox_045_original_b000"
SEEDS = (7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11)  # the first is the page's opening sample
HISTORY = (24, 12)  # control frames back, drawn as ghosts


def record_chain(gen, batch: dict, seed: int) -> list[torch.Tensor]:
    """One sample; the packed (normalised, absolute) output at every state of the reverse chain, k = 10 (the
    prior) ... 0 (the plan). The sampler is left untouched: the two methods its loop calls on the states are
    wrapped for the call (``_featurize`` sees each state before its step, ``_pack_output`` the last)."""
    states = []
    feat, pack = gen._featurize, gen._pack_output

    def featurize(s, t):
        states.append(pack(s))
        return feat(s, t)

    def pack_output(s):
        out = pack(s)
        states.append(out)
        return out

    gen._featurize, gen._pack_output = featurize, pack_output
    try:
        torch.manual_seed(seed)
        gen.generate(batch)
    finally:
        del gen._featurize, gen._pack_output
    return states


def poses(m, d, bodies, traj, i: int) -> np.ndarray:
    """(frames, bodies, 7) for sample ``i`` of a decoded TrajectoryData."""
    rp = traj.body_pos_w[i, :, 0].cpu().numpy()
    rq = traj.body_quat_w[i, :, 0].cpu().numpy()
    jp = traj.joint_pos[i].cpu().numpy()
    op, oq = traj.object_pos_w[i].cpu().numpy(), traj.object_quat_w[i].cpu().numpy()
    out = []
    for f in range(len(rp)):
        q = np.zeros(m.nq)
        q[0:3], q[3:7], q[7:36], q[36:39], q[39:43] = rp[f], rq[f], jp[f], op[f], oq[f]
        out.append(fk(m, d, q, bodies, np.zeros(3)))
    return np.stack(out)


def main() -> None:
    z = np.load(REC, allow_pickle=True)
    m = mujoco.MjModel.from_binary_path(str(REC.parent / json.loads(str(z["meta"]))["scene"]))
    d = mujoco.MjData(m)
    bodies = body_ids(m)
    device = torch.device("cuda:0")
    gens = {k: load_generator_checkpoint(p, device=str(device)).to(device).eval() for k, p in ARMS.items()}
    ref = gens["so3"]
    ds = TrajectoryPairDataset(DATA, split="all", n_val_motions=0, fields=required_raw_fields(ref.repr_config, ref.num_dofs))
    sample = ds[[e["name"] for e in ds._entries].index(MOTION)]
    source = {k: v.to(device) for k, v in sample["source"].items() if torch.is_tensor(v)}
    target = {k: v.to(device) for k, v in sample["target"].items() if torch.is_tensor(v)}
    n_f = int(ref.n_future)

    # The chunk: carried (lifted), where the object turns most within the horizon.
    oq = target["object_rot"].cpu().numpy()
    oz = target["object_pos"][:, 2].cpu().numpy()
    lifted = np.flatnonzero(oz > oz[0] + 0.15)
    dots = np.abs(np.sum(oq[lifted] * oq[np.minimum(lifted + n_f, len(oq) - 1)], axis=1))
    t = int(lifted[np.argmin(dots)])
    print(f"chunk at frame {t}: object turns {np.degrees(2 * np.arccos(dots.min())):.0f} deg over {n_f} frames")

    arms, batches = [], {}
    for key, gen in gens.items():
        enc = MotionEncoder(gen.repr_config, num_dofs=gen.num_dofs)
        norm = gen.norm_stats
        cond = build_sliding_window_condition(
            enc, source_traj=source, state_traj=target, source_frame_idx=t, state_frame_idx=t,
            n_history=int(gen.n_history), n_future=n_f, target_future_traj=target, goal_final_traj=target,
        ).normalized(state_normalizer=norm.state, goal_normalizer=norm.goal,
                     source_normalizer=getattr(norm, "source", None)).as_dict()
        cond = {k: v for k, v in cond.items() if k != "target_future" and not k.startswith("gt_")}
        batches[key] = {k: v.unsqueeze(0).contiguous().to(device) for k, v in cond.items()}
        arms.append(key)

    def chains(seed: int) -> np.ndarray:
        out = []
        for key, gen in gens.items():
            batch = batches[key]
            with torch.no_grad():
                raw = torch.cat(record_chain(gen, batch, seed), dim=0)         # (11, T, D), k = 10 ... 0
                traj = gen.decode_raw_to_trajectory(raw, {k: v.expand(len(raw), *v.shape[1:]) for k, v in batch.items()})
            lev = np.stack([poses(m, d, bodies, traj, i) for i in range(len(raw))])  # (11, T, bodies, 7)
            out.append(lev[::-1])                                              # k = 0 (the plan) ... 10 (the prior)
        return np.stack(out).astype("<f4")                                     # (2, 11, T, bodies, 7)

    q = target
    def state_at(f: int) -> np.ndarray:
        x = np.zeros(m.nq)
        x[0:3], x[3:7], x[7:36] = q["root_pos"][f].cpu(), q["root_rot"][f].cpu(), q["dof_pos"][f].cpu()
        x[36:39], x[39:43] = q["object_pos"][f].cpu(), q["object_rot"][f].cpu()
        return fk(m, d, x, bodies, np.zeros(3))
    now = state_at(t)
    history = np.stack([state_at(max(0, t - h)) for h in HISTORY])
    goal = np.concatenate([q["object_pos"][-1].cpu().numpy(), q["object_rot"][-1].cpu().numpy()]).astype(np.float32)
    for i, seed in enumerate(SEEDS):
        levels = chains(seed)
        if i == 0:
            dump("denoise_cmp", {
                "arms": arms, "checkpoints": [ARMS[a] for a in arms], "motion": MOTION, "frame": t, "seeds": list(SEEDS),
                "samples": len(SEEDS), "levels": levels.shape[1], "frames": levels.shape[2], "bodies": len(bodies),
                "history": len(HISTORY), "goal": goal.round(4).tolist(), "box_half": [0.155, 0.155, 0.17],
            }, levels, now.astype("<f4"), history.astype("<f4"))
        else:
            dump(f"denoise_cmp_{i}", {"seed": seed}, levels)
        print(f"sample {i}: seed {seed}")


if __name__ == "__main__":
    main()
