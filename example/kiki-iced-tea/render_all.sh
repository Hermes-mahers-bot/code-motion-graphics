#!/bin/bash
# Ki Ki ad: render 64s in 4 parallel chunks (aligns with shot boundaries), concat losslessly.
set -e
cd "$(dirname "$0")"
mkdir -p chunks logs
rm -f chunks/*.mp4
for i in 0 1 2 3; do
  S=$((i * 480)); E=$(((i + 1) * 480))
  python3 render.py range $S $E chunks/chunk_$i.mp4 > logs/chunk_$i.log 2>&1 &
done
wait
: > chunks/list.txt
# ABSOLUTE paths: the concat demuxer resolves relative entries against the list file's own dir
for i in 0 1 2 3; do echo "file '$PWD/chunks/chunk_$i.mp4'" >> chunks/list.txt; done
ffmpeg -v error -y -f concat -safe 0 -i chunks/list.txt -c copy video_silent.mp4
echo "--- video ---"
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames -of json video_silent.mp4
