set -euo pipefail

ROOT=~/Downloads/board_dataset_work
VIDDIR="$ROOT/videos"
OUT="$ROOT/dataset"

rm -rf "$OUT"
mkdir -p "$OUT/train/LED_ON" "$OUT/train/LED_OFF" "$OUT/val/LED_ON" "$OUT/val/LED_OFF"

TMP="$ROOT/tmp_frames"
rm -rf "$TMP"; mkdir -p "$TMP"

extract () {
  local f="$1"; local label="$2"
  local stem; stem=$(basename "$f" | sed 's/\..*$//')
  local dump="$TMP/${stem}"
  mkdir -p "$dump"

  ffmpeg -y -loglevel error -i "$f" -vf "fps=2,scale=-1:224" "$dump/${stem}_%05d.jpg"

  find "$dump" -type f -name '*.jpg' | sort -R | awk 'NR%5==0{print > "val.txt"; next} {print > "train.txt"}'
  while IFS= read -r p; do cp "$p" "$OUT/train/$label/"; done < train.txt
  while IFS= read -r p; do cp "$p" "$OUT/val/$label/"; done < val.txt
  rm -f train.txt val.txt
}

for f in "$VIDDIR"/*.MOV; do
  base=$(basename "$f")
  shopt -s nocasematch
  if [[ "$base" == Lighton* ]]; then
    extract "$f" "LED_ON"
  elif [[ "$base" == Lightoff* ]]; then
    extract "$f" "LED_OFF"
  else
    echo "Skip unknown file: $f"
  fi
done

echo "Done. Dataset at: $OUT"
