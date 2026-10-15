#!/usr/bin/env bash
# Downloads Stroebl et al. (2024), figshare 25975315 v1. Same-name files are kept apart by adding figshare's file id.
set -euo pipefail
OUT="data/raw/stroebl"
rm -rf "$OUT" && mkdir -p "$OUT"                     # start clean
curl -s https://api.figshare.com/v2/articles/25975315 > "$OUT/_metadata.json"

jq -r '.files[] | "\(.id)\t\(.name)\t\(.download_url)\t\(.computed_md5)"' "$OUT/_metadata.json" |
while IFS=$'\t' read -r id name url md5; do
  dest="$OUT/${name%.*}__${id}.${name##*.}"          # e.g. TP_k01_02__47627056.zip (keeps the extension)
  curl -sL -o "$dest" "$url"                         # download (no resume, so no mixed-up files)
  [ "$(md5 -q "$dest")" = "$md5" ] && echo "ok  $dest" || echo "BAD $dest"   # verify immediately
done