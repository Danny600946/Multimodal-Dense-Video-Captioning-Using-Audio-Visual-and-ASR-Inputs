# File: demo_audio_features.py
# Author: Daniel Vousden
#
# Description:
# This script extracts raw audio features from video files using Facebook's Wav2Vec 2.0 model.
# It processes audio into fixed-size embeddings suitable for integration with a dense video captioning pipeline.
# The script performs the following steps:
# - Extracts audio from a video using ffmpeg.
# - Segments the audio and feeds it through a pretrained Wav2Vec model.
# - Downsamples the high-resolution features to a fixed number of steps (default: 100).
# - Saves the resulting tensor as a .pt file.
#
# Key Components:
# - Uses torchaudio for waveform resampling.
# - Uses Hugging Face Transformers for feature extraction.
# - Designed to support batch processing of videos into reusable audio feature files.
#
# License: Assumes fair use under academic MIT/open-source conditions (Wav2Vec2: Facebook AI, MIT License).


import torchaudio
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import ffmpeg
import sys

# Extract audio from video
def extract_audio_from_video(video_path, wav_out_path="temp_audio.wav"):
    ffmpeg.input(video_path).output(wav_out_path, ac=1, ar=16000).run(overwrite_output=True)
    return wav_out_path

# Load model
def load_wav2vec_model():
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()
    return processor, model

# Segmented audio feature extraction
def extract_features(audio_path, processor, model, device="cpu", chunk_size_sec=10):
    model.to(device)

    waveform, sample_rate = sf.read(audio_path)
    waveform = torch.tensor(waveform).float().unsqueeze(0) 

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    total_samples = waveform.shape[-1]
    chunk_size = chunk_size_sec * 16000 
    features = []

    with torch.no_grad():
        for i in range(0, total_samples, chunk_size):
            chunk = waveform[:, i:i+chunk_size]
            if chunk.shape[-1] < chunk_size:
                chunk = F.pad(chunk, (0, chunk_size - chunk.shape[-1]))  

            inputs = processor(chunk.squeeze(), sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(device)
        
            output = model(input_values).last_hidden_state.squeeze(0)[:, :768]
            features.append(output.cpu()) 
    # combines segments
    return torch.cat(features, dim=0) 
# Reduces feature length to 100 by using mean pooling
def downsample_features(features, target_steps=100):
    total_steps = features.size(0)
    segment_size = total_steps // target_steps

    pooled = []
    for i in range(target_steps):
        start = i * segment_size
        end = (i + 1) * segment_size if i < target_steps - 1 else total_steps
        segment = features[start:end]
        pooled.append(segment.mean(dim=0)) 

    return torch.stack(pooled) 


def save_features(features, out_path="audio_features.pt"):
    torch.save(features, out_path)
# CLI interface: python extract_audio_features.py your_video.mp
# Extracts audio feature form the provided video
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_path = sys.argv[1] if len(sys.argv) > 1 else "your_video.mp4"
    audio_path = extract_audio_from_video(video_path)
    processor, model = load_wav2vec_model()
    features = extract_features(audio_path, processor, model, device)
    features = downsample_features(features, target_steps=100)
    save_features(features, "audio_features.pt")

    print("Extracted Wav2Vec features:", features.shape)