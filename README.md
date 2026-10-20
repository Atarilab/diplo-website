# DiPLO project page

Static site, no build tooling beyond Python 3 and ffmpeg.

```
index.html              the page to host (generated from src/page.html)
src/page.html           source: styles, markup and script in one file
static/videos/*.mp4     web encodes of the shots in zerosho/outputs/video (720p25, grids 1080p)
static/posters/*.jpg    one poster per clip
static/img/og.jpg       the 1200x630 link-preview image
static/data/            data behind the interactive visuals (G1 meshes, a recorded rollout and its plans, one plan
                        at ten noise levels, the SBTO paths); made by tools/export_web_data.py
tools/build_media.sh    re-encode clips and posters:  bash tools/build_media.sh [id ...]
tools/export_web_data.py  cd ../zerosho && uv run --no-sync --with fast-simplification python ../diplo-website/tools/export_web_data.py
tools/export_goal_map.py  python3 tools/export_goal_map.py   (goal-grid error maps from zerosho/eval_results/radial_goal_study)
tools/export_replays.py cd ../zerosho && uv run --no-sync --with fast-simplification python ../diplo-website/tools/export_replays.py
                        (the 3D replay's six rollouts, replay.json first, from outputs/video/rollouts/pp29x8_final.npz
                        and pp29_final.npz: the least staggering successful rollout of six walking carries)
tools/build_index.py    src/page.html -> index.html:  python3 tools/build_index.py --url https://<site>/
```

## Publishing

The site is served by GitHub Pages from the root of `main` at https://atarilab.github.io/diplo-website/.
After editing `src/page.html`, rebuild and push:

```
python3 tools/build_index.py --url https://atarilab.github.io/diplo-website/
git add -A && git commit -m "..." && git push
```

`LINKS = { paper, arxiv, code }` at the top of the script in `src/page.html` sets the hero buttons; an empty one
shows as "soon".

## Preview locally

```
python3 -m http.server 8000   # then open http://localhost:8000
```

## The film

The film plays from YouTube (https://youtu.be/rDrdnYQUYoM): the page shows the local poster
(`static/posters/diplo_video.jpg`) and loads the youtube-nocookie player in place only on click. Only where the host's
content policy blocks YouTube frames (the private Artifact preview does) does it play `static/videos/diplo_video.mp4`
instead; that 38 MB file is git-ignored and not published on GitHub Pages.

`.gitignore` also keeps out the media and data of sections that were removed from the page.

## Interactive visuals

The method section and the 3D replay under it draw the robot with three.js (r128, from cdnjs/jsdelivr) from the
recorded simulation poses in `static/data/`; the SD:DPO and DA:PPO diagrams are schematics drawn in SVG. Without
WebGL the viewers fall back to a still. The page runs film → method (four tabs, the task-conditioning chart over
the `priors` strip video, the 3D replay) → post-training results (generalization to new goals, other skills) → real
world → BibTeX.

## Sources of the media

All clips come from the video project in `zerosho/outputs/video` (the `rough_cut_v26` shots); the clip table
at the top of `tools/build_media.sh` maps each id to its source and trim. Numbers on the page are from the paper
(`IEEE_2026_DiffRL/main.tex`).

The strip under the task-prior chart (`priors`: one panel per prior, one human motion as each prior gives it) is
rendered in the zerosho repo, then encoded here:

```
uv run python scripts/video/render_priors.py out=outputs/video/shots/web_priors/priors_strip.mp4   # in zerosho
bash tools/build_media.sh priors
```
