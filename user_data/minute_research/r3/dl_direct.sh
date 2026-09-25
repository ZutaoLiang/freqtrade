#!/bin/bash
# usage: dl.sh listfile rootdir parallel ; each line "url relpath"; skips existing, records 404s
list=$1; root=$2; P=$3
mkdir -p $root
cat $list | xargs -P $P -L 1 bash -c 'u=$0; p='"$root"'/$1; [ -s "$p" ] && exit 0; mkdir -p $(dirname $p); code=$(curl -s --noproxy "*" -o "$p.tmp" -w "%{http_code}" --retry 3 --max-time 120 "$u"); if [ "$code" = 200 ]; then mv "$p.tmp" "$p"; else rm -f "$p.tmp"; echo "$code $u" >> '"$root"'/missing.txt; fi'
echo DONE
