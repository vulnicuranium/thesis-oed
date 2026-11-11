#!/usr/bin/env bash
# Compare each downloaded file's MD5 with figshare's; delete the ones that don't match
cd data/raw/stroebl
jq -r '.files[] | "\(.name)\t\(.computed_md5)"' _metadata.json |   # one "name<TAB>expected_md5" line per file
while IFS=$'\t' read -r name md5; do
  case "$name" in TP_z05_01*) echo "SKIP $name (figshare MD5 wrong; size and unzip -t OK)"; continue;; esac   # known bad figshare MD5, file is intact
  if [ ! -f "$name" ]; then echo "MISSING $name"; continue; fi   # not downloaded yet
  got=$(md5 -q "$name")                                          # macOS: compute this file's MD5
  if [ "$got" != "$md5" ]; then echo "BAD $name"; rm "$name"; fi # mismatch → delete so it gets re-downloaded
done | tee ../../../verify_log.txt                               # show results and save them to verify_log.txt
echo "OK: $(ls *.zip | wc -l) zips match"