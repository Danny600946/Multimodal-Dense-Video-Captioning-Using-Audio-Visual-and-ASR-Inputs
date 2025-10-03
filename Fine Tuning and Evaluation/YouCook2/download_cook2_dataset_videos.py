# File: download_cook2_dataset_videos.py
# Author: Daniel Vousden
# Description: Downloads YouCook2 dataset videos from youtube.
# This file is part of my final year dissertation for the University of Sheffield (2025).

import os
import json
import subprocess

DATASET_JSON_PATH = "val.json"  
SAVE_VIDEOS_DIR = "youcook2_val_videos"  
SUCCESSFUL_IDS_FILE = "successful_ids.txt"  
YTDLP_PATH = "yt-dlp"  
NUM_VIDEOS_LIMIT = None  

# Create save folder if it doesn't exist
os.makedirs(SAVE_VIDEOS_DIR, exist_ok=True)

# Load video IDs from the JSON
with open(DATASET_JSON_PATH, "r") as f:
    data = json.load(f)

video_ids = list(data.keys())

if NUM_VIDEOS_LIMIT:
    video_ids = video_ids[:NUM_VIDEOS_LIMIT]

print(f"[INFO] Attempting to download {len(video_ids)} videos...")

successful_ids = []

for vid_id in video_ids:
    save_path = os.path.join(SAVE_VIDEOS_DIR, f"{vid_id}.mp4")

    # Skips download if found locally
    if os.path.exists(save_path):
        print(f"[SKIP] {vid_id} already downloaded.")
        successful_ids.append(vid_id)
        continue

    youtube_url = f"https://www.youtube.com/watch?v={vid_id}"
    print(f"[DOWNLOAD] {youtube_url}")

    try:
        # Download using yt-dlp
        subprocess.run([
            YTDLP_PATH,
            "-f", "best[ext=mp4]/best",
            "--output", save_path,
            youtube_url
        ], check=True)

        print(f"[SUCCESS] Downloaded {vid_id}")
        successful_ids.append(vid_id)
    # Skips video if download fails
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to download {vid_id}, skipping.")

# Saves a file of the successfully downloaded videos
with open(SUCCESSFUL_IDS_FILE, "w") as f:
    for vid_id in successful_ids:
        f.write(vid_id + "\n")

print(f"[DONE] Downloaded {len(successful_ids)} videos successfully.")
print(f"[INFO] Saved successful IDs to {SUCCESSFUL_IDS_FILE}")
