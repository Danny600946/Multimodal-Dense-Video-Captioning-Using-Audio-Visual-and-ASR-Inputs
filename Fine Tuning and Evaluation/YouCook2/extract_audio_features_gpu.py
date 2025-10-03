# File: extract_audio_features_gpu.py
# Author: Daniel Vousden
# Description: Implements audio feature extraction and mean pooling from Wav2Vec for use in multimodal captioning.
# This file is part of my final year dissertation for the University of Sheffield (2025).

import os
import torch
import soundfile as sf
import torchaudio
import ffmpeg
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

video_folder = "YouCook2/youcook2_train_videos/"
output_folder = "YouCook2/audio_features_train_768/"
# Makes directories if they don't exist
os.makedirs(output_folder, exist_ok=True)

# Load Wav2Vec2-BASE model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
model.eval()

# Extracts the audio from the video
def extract_audio(video_path, wav_path="temp_audio.wav"):
    ffmpeg.input(video_path).output(wav_path, ac=1, ar=16000).overwrite_output().run(quiet=True)
    return wav_path

# Loads audio and extracts features.
def extract_raw_features(wav_path):
    waveform, sample_rate = sf.read(wav_path)
    waveform = torch.tensor(waveform).float().unsqueeze(0) 
    # Sample audio to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    # Disables gradient computation.
    with torch.no_grad():
        # Normalises tensors and extracts features using wave2vec.
        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        outputs = model(input_values).last_hidden_state.squeeze(0)

    return outputs.cpu()

# Converts all feature vectors to a total of 100.
def downsample_to_100_steps(features, target_steps=100):
    # Get total timesteps and calc segment size.
    total_steps = features.size(0)
    segment_size = total_steps // target_steps

    pooled_features = []
    # Calc the mean pool over the time segment.
    for i in range(target_steps):
        start = i * segment_size
        #  The rest of the steps are given to the last segment.
        end = (i + 1) * segment_size if i < target_steps - 1 else total_steps
        segment = features[start:end]
        pooled_features.append(segment.mean(dim=0))  

    return torch.stack(pooled_features)  

# Iterates over all .mp4 files extracting their features in the shape [1, 100, 768].
if __name__ == "__main__":
    # Gets video from the folder.
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    print(f"[INFO] Found {len(video_files)} videos. Starting extraction...")
    # Tracks extraction progress
    progress_bar = tqdm(video_files, ncols=100, dynamic_ncols=True, smoothing=0.1)
    
    for video_file in progress_bar:
        # Makes output paths.
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_folder, video_file)
        output_path_trimmed = os.path.join(output_folder, f"{video_id}.pt")

        # Skip if feature file already exists
        if os.path.exists(output_path_trimmed):
            continue
        # Extracts features
        wav_path = extract_audio(video_path)
        raw_features = extract_raw_features(wav_path)
        features_100x768 = downsample_to_100_steps(raw_features, target_steps=100)
        torch.save(features_100x768, output_path_trimmed)
        # Deletes temporary .wav for space
        os.remove(wav_path)

    print("[INFO] Audio feature extraction complete.")
