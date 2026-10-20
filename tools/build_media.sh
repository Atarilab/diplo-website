#!/usr/bin/env bash
# Re-encode the DiPLO video shots for the web: H.264 MP4 (faststart, 720p25, no audio unless kept)
# plus one JPEG poster per clip. Sources are the renders in zerosho/outputs/video.
#
#   bash tools/build_media.sh            # all clips + the full video
#   bash tools/build_media.sh plan sdpo  # only these ids ("cz" = all carousel slides, "film" = the full video)
set -euo pipefail

SRC=${SRC:-/home/breakout/workspace/zerosho/outputs/video}
HERE=$(cd "$(dirname "$0")/.." && pwd)
VID="$HERE/static/videos"
POS="$HERE/static/posters"
mkdir -p "$VID" "$POS"

# id | source (relative to $SRC) | trim start s | trim end s ("" = to the end) | poster time s (in the trimmed clip) | audio (keep|none)
# Grids of small tiles (and the seven-panel priors strip) keep 1920 px width (GRIDS) so they stay readable when expanded; the rest are 1280.
CLIPS=(
  "hero_twin|shots/o1_twin_open_v5.mp4|5.0|13.8|1.6|none"
  "data|shots/m1_data_sub7_045_v5.mp4|0||17.5|none"
  "pretrain|shots/m_pretrain_v2.mp4|0||9.0|none"
  "sdpo|shots/m_sdpo_v2.mp4|0||10.5|none"
  "dappo|shots/m_dappo.mp4|0||10.0|none"
  "plan|shots/s02_plan_sub7_045.mp4|0||4.2|none"
  "prog_pick|shots/s05_progression_sub3_004_v2.mp4|0||8.6|none"
  "prog_turn|shots/s05_progression_sub10_040_v2.mp4|0||6.6|none"
  "baselines|shots/s06_baselines_stacked_v5.mp4|0||11.0|none"
  "dppo_sim|shots/s11_dppo_cmp_sub3_003_junetf_xy_v5.mp4|0||7.5|none"
  "dppo_real|shots/s11b_real_dppo_cmp_v3.mp4|0||4.0|keep"
  "fanout|shots/s07_fanout_sub3_003.mp4|0||10.5|none"
  "moved|shots/s07b_fig5_moved_sub10_084.mp4|0||9.5|none"
  "turned|shots/s07c_fig5_rotated_sub3_005_v2.mp4|0||11.0|none"
  "movegoal_grid|shots/s09_movegoal_grid_v3.mp4|0||9.0|none"
  "far20|shots/s09b_far20_sub3_003_v2.mp4|0||6.0|none"
  "tasks_grid|shots/s08_tasks_grid_v3_alt.mp4|0||5.0|none"
  "real_rep_grid|real/rep_grid.mp4|0||5.0|none"
  "real_carry|real/dist_carry.mp4|0||5.0|none"
  "real_turn_lift|real/dist_turn_lift.mp4|0||4.4|none"
  "real_rotate|real/dist_rotate.mp4|0||4.0|none"
  "priors|shots/web_priors/priors_strip.mp4|0||3.2|none"
)

want() {  # no ids given = everything
  [[ $# -eq 0 ]] && return 0
  local id=$1; shift
  for w in "${ONLY[@]}"; do [[ $w == "$id" ]] && return 0; done
  return 1
}
ONLY=("$@")
GRIDS=" baselines movegoal_grid tasks_grid real_rep_grid priors "

for row in "${CLIPS[@]}"; do
  IFS='|' read -r id src t0 t1 tp audio <<<"$row"
  if [[ ${#ONLY[@]} -gt 0 ]] && ! want "$id" "${ONLY[@]}"; then continue; fi
  trim=(-ss "$t0")
  [[ -n $t1 ]] && trim+=(-to "$t1")
  if [[ $audio == keep ]]; then a=(-c:a aac -b:a 96k); else a=(-an); fi
  if [[ $GRIDS == *" $id "* ]]; then w=1920; else w=1280; fi
  ffmpeg -v error -y "${trim[@]}" -i "$SRC/$src" \
    -vf "fps=25,scale=$w:-2:flags=lanczos" -c:v libx264 -preset slow -crf 25 -profile:v high \
    -pix_fmt yuv420p -movflags +faststart "${a[@]}" "$VID/$id.mp4"
  ffmpeg -v error -y -ss "$tp" -i "$VID/$id.mp4" -frames:v 1 -vf "scale=1280:-2" -q:v 4 "$POS/$id.jpg"
  printf '%-16s %6s KB\n' "$id" "$(( $(stat -c %s "$VID/$id.mp4") / 1024 ))"
done

# Carousel slides: one clip per tile of the grid shots, full size (zerosho/outputs/video/rollouts/render_web_tiles.sh
# renders the simulated ones; the real ones are real_footage.py's takes). Each gets a 1280 px poster and a 320 px thumb.
# id | source (relative to $SRC) | poster time s
CZ=(
  "real_01|real/rep_01.mp4|5.0" "real_02|real/rep_02.mp4|5.0" "real_03|real/rep_03.mp4|5.0" "real_04|real/rep_04.mp4|5.0"
  "real_05|real/rep_05.mp4|5.0" "real_06|real/rep_06.mp4|5.0" "real_07|real/rep_07.mp4|5.0" "real_08|real/rep_08.mp4|5.0"
  "real_09|real/rep_09.mp4|5.0" "real_10|real/rep_11.mp4|5.0"
  "mg_env0|shots/web_tiles/mg_env0.mp4|7.5" "mg_env4|shots/web_tiles/mg_env4.mp4|7.5" "mg_env9|shots/web_tiles/mg_env9.mp4|7.5"
  "mg_back180_env1|shots/web_tiles/mg_back180_env1.mp4|7.5" "mg_env16|shots/web_tiles/mg_env16.mp4|7.5"
  "mg_env23|shots/web_tiles/mg_env23.mp4|7.5" "mg_env40|shots/web_tiles/mg_env40.mp4|7.5" "mg_env46|shots/web_tiles/mg_env46.mp4|7.5"
  "mg_env26|shots/web_tiles/mg_env26.mp4|7.5" "mg_env29|shots/web_tiles/mg_env29.mp4|7.5"
  "mg_zigzag_b_env10|shots/web_tiles/mg_zigzag_b_env10.mp4|8.0" "mg_env39|shots/web_tiles/mg_env39.mp4|7.5"
  "task_kick_env10|shots/web_tiles/task_kick_env10.mp4|5.0" "task_kick_env46|shots/web_tiles/task_kick_env46.mp4|5.0"
  "task_kick_env52|shots/web_tiles/task_kick_env52.mp4|5.0" "task_kick_env30|shots/web_tiles/task_kick_env30.mp4|5.0"
  "task_pushpull_env23|shots/web_tiles/task_pushpull_env23.mp4|6.0" "task_pushpull_env11|shots/web_tiles/task_pushpull_env11.mp4|6.0"
  "task_pushpull_env25|shots/web_tiles/task_pushpull_env25.mp4|6.0" "task_pushpull_env12|shots/web_tiles/task_pushpull_env12.mp4|6.0"
  "base_sub10_088|shots/web_tiles/base_sub10_088.mp4|8.0" "base_sub3_004|shots/web_tiles/base_sub3_004.mp4|8.0"
)
THUMBS_FROM_POSTERS=(real_carry real_turn_lift real_rotate)  # slides that reuse a clip above
mkdir -p "$VID/cz" "$POS/cz" "$HERE/static/thumbs"
for row in "${CZ[@]}"; do
  IFS='|' read -r id src tp <<<"$row"
  if [[ ${#ONLY[@]} -gt 0 ]] && ! want "$id" "${ONLY[@]}" && ! want cz "${ONLY[@]}"; then continue; fi
  w=1280; [[ $id == base_* ]] && w=1920
  ffmpeg -v error -y -i "$SRC/$src" -vf "fps=25,scale=$w:-2:flags=lanczos" -c:v libx264 -preset slow -crf 25 \
    -profile:v high -pix_fmt yuv420p -movflags +faststart -an "$VID/cz/$id.mp4"
  ffmpeg -v error -y -ss "$tp" -i "$VID/cz/$id.mp4" -frames:v 1 -vf "scale=1280:-2" -q:v 4 "$POS/cz/$id.jpg"
  ffmpeg -v error -y -i "$POS/cz/$id.jpg" -vf "scale=320:-2" -q:v 5 "$HERE/static/thumbs/$id.jpg"
  printf '%-22s %6s KB\n' "cz/$id" "$(( $(stat -c %s "$VID/cz/$id.mp4") / 1024 ))"
done
if [[ ${#ONLY[@]} -eq 0 ]] || want cz "${ONLY[@]}"; then
  for id in "${THUMBS_FROM_POSTERS[@]}"; do
    ffmpeg -v error -y -i "$POS/$id.jpg" -vf "scale=320:-2" -q:v 5 "$HERE/static/thumbs/$id.jpg"
  done
fi

# The full narrated video: 1080p for the deployed site, a <=15 MB copy for the private Artifact preview.
if [[ ${#ONLY[@]} -eq 0 ]] || want film "${ONLY[@]}"; then
  FILM="$SRC/cuts/rough_cut_v26_narrated.mp4"
  ffmpeg -v error -y -i "$FILM" -c:v libx264 -preset slow -crf 24 -profile:v high -pix_fmt yuv420p \
    -r 25 -c:a aac -b:a 128k -movflags +faststart "$VID/diplo_video.mp4"
  mkdir -p "$HERE/build/preview"
  ffmpeg -v error -y -i "$FILM" -vf "fps=25,scale=1280:-2:flags=lanczos" -c:v libx264 -preset slow \
    -b:v 430k -maxrate 900k -bufsize 1800k -pix_fmt yuv420p -c:a aac -b:a 64k -movflags +faststart \
    "$HERE/build/preview/diplo_video.mp4"
  ffmpeg -v error -y -ss 4.2 -i "$FILM" -frames:v 1 -vf "scale=1280:-2" -q:v 3 "$POS/diplo_video.jpg"
  printf '%-16s %6s KB (site)  %6s KB (preview)\n' film \
    "$(( $(stat -c %s "$VID/diplo_video.mp4") / 1024 ))" "$(( $(stat -c %s "$HERE/build/preview/diplo_video.mp4") / 1024 ))"
fi
