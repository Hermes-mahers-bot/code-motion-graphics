#!/bin/bash
# Mux the silent render with the composed audio bed, level it cleanly, then make delivery copies.
set -e
cd "$(dirname "$0")"

# 1) master: video + audio, loudness-normalised so lossy AAC never clips
ffmpeg -v error -y -i video_silent.mp4 -i mix/audio.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -ar 44100 -ac 2 \
  -af "loudnorm=I=-16:TP=-2.5:LRA=9" -movflags +faststart -shortest kiki_ad.mp4

ffmpeg -v error -y -ss 9.2 -i kiki_ad.mp4 -frames:v 1 -q:v 3 poster.jpg

# 2) delivery copy: 1080p re-encode at a chat-friendly bitrate
ffmpeg -v error -y -i kiki_ad.mp4 -c:v libx264 -preset medium -crf 25 \
  -pix_fmt yuv420p -c:a copy -movflags +faststart kiki_ad_deliver.mp4

# 3) small fallback
ffmpeg -v error -y -i kiki_ad.mp4 -vf "scale=1280:-2:flags=lanczos" \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a copy \
  -movflags +faststart kiki_ad_720p.mp4

echo "--- master ---"
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,codec_type,width,height,avg_frame_rate,nb_frames \
  -of default=nw=1 kiki_ad.mp4
echo "--- levels ---"
ffmpeg -v info -i kiki_ad.mp4 -af volumedetect -f null - 2>&1 | grep -E "max_volume|mean_volume"
ls -la kiki_ad.mp4 kiki_ad_deliver.mp4 kiki_ad_720p.mp4 poster.jpg
