"""Goal-grid data for the page's reach map: the radial goal study of the zerosho repo.

Goals on a polar grid around the robot's start (8 radii from 0.5 to 5 m, a bearing every 30 deg, four goal yaws,
32 rollouts each), for the three checkpoints of the paper's Fig. 9 and Appendix M: the pretrained student, + SD:DPO
and + SD:DPO + DA:PPO. Per goal position (four yaws pooled, 128 rollouts) it keeps the share of rollouts that end
with the object within 20 cm of the goal without a fall (the paper's 88 %), and the mean final position and
orientation errors over the rollouts that lifted the object (the paper's maps). Stdlib only:

    python3 tools/export_goal_map.py
"""

import csv
import json
import os

STUDY = os.path.join(os.path.dirname(__file__), "..", "..", "zerosho", "eval_results", "radial_goal_study")
OUT = os.path.join(os.path.dirname(__file__), "..", "static", "data", "goal_map.json")
MODELS = [("student", "models/student/rollouts"), ("sdpo", "models/student_sdpo/rollouts"), ("sdda", "rollouts")]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def stats(rows):
    n = len(rows)
    reach = sum(float(r["pos_err_m"]) <= 0.20 and r["survived"] == "1" for r in rows)
    lifted = [r for r in rows if r["lifted"] == "1"]
    pos = sum(float(r["pos_err_m"]) for r in lifted) / max(1, len(lifted))
    ori = sum(float(r["ori_err_d4_deg"]) for r in lifted) / max(1, len(lifted))
    alive = sum(r["survived"] == "1" for r in rows)
    return n, reach, len(lifted), round(pos, 4), round(ori, 2), alive


def main():
    out = {"radii": None, "bearings": None, "models": {}, "demo": []}
    for key, sub in MODELS:
        grid = read(os.path.join(STUDY, sub, "c1.75_weighted.csv"))
        demo = read(os.path.join(STUDY, sub, "c1.75_dataset_weighted.csv"))
        radii = sorted({float(r["radius_m"]) for r in grid})
        bearings = sorted({float(r["bearing_deg"]) for r in grid})
        out["radii"], out["bearings"] = radii, bearings
        cells = []
        for rad in radii:
            for b in bearings:
                cells.append(stats([r for r in grid if float(r["radius_m"]) == rad and float(r["bearing_deg"]) == b]))
        out["models"][key] = {"cells": cells, "grid": stats(grid), "demo": stats(demo)}
        if key == "sdda":
            seen = set()
            for r in demo:
                if r["motion"] in seen:
                    continue
                seen.add(r["motion"])
                out["demo"].append([round(float(r["goal_x"]), 3), round(float(r["goal_y"]), 3), round(float(r["goal_yaw_deg"]), 1)])
    out["fields"] = ["rollouts", "within_20cm", "lifted", "pos_err_lifted_m", "ori_err_lifted_deg", "survived"]
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    for key, m in out["models"].items():
        n, reach, lifted, pos, ori, alive = m["grid"]
        print(f"{key}: within 20 cm {100 * reach / n:.1f} %, survival {100 * alive / n:.1f} %, lifted {100 * lifted / n:.1f} %, "
              f"pos {pos:.3f} m, ori {ori:.1f} deg  (demo goals {100 * m['demo'][1] / m['demo'][0]:.1f} %)")
    print(OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
