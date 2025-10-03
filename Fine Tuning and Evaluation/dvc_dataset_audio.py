# File: dvc_dataset_audio.py
# Original Author(s): VidChapters Team (https://github.com/antoyang/VidChapters)
# Modified by: Daniel Vousden
#
# Description:
# This file extends the original VidChapters dataset loading and collation logic to include audio features
# as an additional input modality. It defines a custom dataset class `DenseVideoCaptioning_Dataset_Audio`
# and a new `collate_fn` to handle batched audio features alongside video, subtitle, and token data.
#
# Key Modifications:
# - Added support for loading audio features from disk (e.g., mean pooled Wav2Vec outputs stored as .pt files).
# - Injected audio features into the `__getitem__` return dict for use in multimodal training.
# - Adapted the `densevideocaptioning_collate_fn_audio` function to stack and batch audio features.
# - Updated `build_densevideocaptioning_dataset_audio` to route correct paths and handle audio parameters.
# - Added safeguards to handle missing audio features gracefully (zero-filled fallback).
#
# License: MIT 

import os
import torch as th
from torch.utils.data import Dataset
import json
import pickle
import numpy as np
from util.t5 import create_sentinel_ids, filter_input_ids, random_spans_noise_mask

class DenseVideoCaptioning_Dataset_Audio(Dataset):
    def __init__(
        self,
        json_path,
        features_path,
        audio_features_path=None,  # Path for audio features
        max_feats=100,
        features_dim=768,
        tokenizer=None,
        subtitles_path=None,
        num_bins=100,
        max_input_tokens=1000,
        max_output_tokens=256,
        noise_density=0.25,
        mean_noise_span_length=5,
    ):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())
        self.features = None
        self.features_path = None
        if os.path.isdir(features_path):
            self.features_path = features_path
        else:
            self.features = th.load(features_path)
        # Store path for audio features
        self.audio_features_path = audio_features_path
        
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.tokenizer = tokenizer
        self.subs = None
        self.subs_path = None
        if subtitles_path and os.path.exists(subtitles_path) and os.path.isdir(subtitles_path):
            self.subs_path = subtitles_path
        elif subtitles_path and os.path.exists(subtitles_path):
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            print("No subtitles given or found.")
        self.num_bins = num_bins
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.num_text_tokens = len(tokenizer) - num_bins
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

    def __len__(self):
        return len(self.data)

    def _get_text(self, text):
        text = text.strip()
        text = text.capitalize()
        if text and text[-1] != '.':
            text = text + '.'
        return text

    def _get_video(self, video_id):
        if self.features is not None:
            assert video_id in self.features, video_id
            video = self.features[video_id].float()
        else:
            features_path = os.path.join(self.features_path, video_id + '.mp4.npy')
            if not os.path.exists(features_path):
                features_path = os.path.join(self.features_path, video_id + '.npy')
            assert os.path.exists(features_path), features_path
            video = th.from_numpy(np.load(features_path)).float()

        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
        elif len(video) < self.max_feats:
            video = th.cat(
                [video, th.zeros(self.max_feats - len(video), self.features_dim)], 0
            )
        return video

    def __getitem__(self, idx):
        video_id = self.vids[idx]
        annotations = self.data[video_id]
        video = self._get_video(video_id[-11:])
        duration = annotations["duration"]

        # Load audio features
        audio_path = os.path.join(self.audio_features_path, f"{video_id}.pt")
        if os.path.exists(audio_path):
            audio_features = th.load(audio_path).float()
        # Uses 0 tensor if audio is missing
        else:
            print(f"[WARN] Audio feature not found for {video_id}. Using zero tensor.")
            # Shape must match normal audio feature shape
            audio_features = th.zeros((100, 768))  


        # get subtitles (unchanged)
        if (self.subs is not None and video_id[-11:] in self.subs) or (self.subs_path is not None and os.path.exists(os.path.join(self.subs_path, video_id + '.pkl'))):
            if (self.subs is not None and video_id[-11:] in self.subs):
                sub = self.subs[video_id[-11:]]
            else:
                sub = pickle.load(open(os.path.join(self.subs_path, video_id[-11:] + '.pkl'), 'rb'))

            to_keep = [(x >= 0 and y <= duration) for x, y in zip(sub["start"], sub["end"])]
            if not any(to_keep):
                input_tokens = (th.ones(1) * self.tokenizer.eos_token_id).long()
            else:
                sub["start"] = [x for i, x in enumerate(sub["start"]) if to_keep[i]]
                sub["end"] = [x for i, x in enumerate(sub["end"]) if to_keep[i]]
                sub['text'] = [self._get_text(x) for i, x in enumerate(sub['text']) if to_keep[i]]
                time_input_tokens = [th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                                    self.time_tokenize(ed, duration, self.num_bins)])
                                     for st, ed in zip(sub['start'], sub['end'])]
                text_input_tokens = [self.tokenizer(x, add_special_tokens=False, max_length=self.max_input_tokens,
                                                    padding="do_not_pad", truncation=True, return_tensors="pt")['input_ids'][0]
                                     for x in sub['text']]
                input_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_input_tokens, text_input_tokens)]
                input_tokens = th.cat(input_tokens, 0)
                input_tokens = input_tokens[:self.max_input_tokens - 1]
                input_tokens = th.cat([input_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)
        else:
            input_tokens = (th.ones(1) * self.tokenizer.eos_token_id).long()

        # denoising sequence (unchanged)
        if len(input_tokens) > 1:
            mask_indices = np.asarray(
                [random_spans_noise_mask(len(input_tokens), self.noise_density, self.mean_noise_span_length)])
            labels_mask = ~mask_indices

            input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), self.tokenizer, self.num_bins)
            labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), self.tokenizer, self.num_bins)

            denoising_output_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), labels_sentinel, self.tokenizer)).squeeze(0)
            denoising_input_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), input_ids_sentinel, self.tokenizer)).squeeze(0)
        else:
            input_tokens = th.LongTensor([self.tokenizer.eos_token_id])
            denoising_input_tokens = th.LongTensor([0])
            denoising_output_tokens = input_tokens

        # dense video caption output tokens
        captions = [self._get_text(x) for x in annotations['sentences']]
        time_output_tokens = [th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                             self.time_tokenize(ed, duration, self.num_bins)])
                              for st, ed in annotations['timestamps']]
        text_output_tokens = [self.tokenizer(x, add_special_tokens=False, max_length=self.max_output_tokens,
                                             padding="do_not_pad", truncation=True, return_tensors="pt")['input_ids'][0]
                              for x in captions]
        output_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_output_tokens, text_output_tokens)]
        output_tokens = th.cat(output_tokens, 0)
        output_tokens = output_tokens[:self.max_output_tokens - 1]
        output_tokens = th.cat([output_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)

        return {
            "video_id": video_id,
            "duration": duration,
            "video": video,
            "audio": audio_features,  # return audio dictionary
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "denoising_input_tokens": denoising_input_tokens,
            "denoising_output_tokens": denoising_output_tokens,
        }

    def time_tokenize(self, x, duration, num_bins):
        time_token = int(float((num_bins - 1) * x) / float(duration))
        assert time_token <= self.num_bins
        return time_token + self.num_text_tokens


def densevideocaptioning_collate_fn_audio(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    duration = [batch[i]["duration"] for i in range(bs)]
    video = th.stack([batch[i]["video"] for i in range(bs)])
    # Batch audio features across samples.
    audio = th.stack([batch[i]["audio"] for i in range(bs)]) 

    # process text tokens 
    input_tokens = [batch[i]["input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in input_tokens)
    for i in range(bs):
        if len(input_tokens[i]) < max_input_len:
            input_tokens[i] = th.cat([input_tokens[i], th.zeros(max_input_len - len(input_tokens[i])).long()], 0)
    input_tokens = th.stack(input_tokens)

    output_tokens = [batch[i]["output_tokens"] for i in range(bs)]
    max_output_len = max(len(x) for x in output_tokens)
    for i in range(bs):
        if len(output_tokens[i]) < max_output_len:
            output_tokens[i] = th.cat([output_tokens[i], th.zeros(max_output_len - len(output_tokens[i])).long()], 0)
    output_tokens = th.stack(output_tokens)

    denoising_input_tokens = [batch[i]["denoising_input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in denoising_input_tokens)
    for i in range(bs):
        if len(denoising_input_tokens[i]) < max_input_len:
            denoising_input_tokens[i] = th.cat([denoising_input_tokens[i], th.zeros(max_input_len - len(denoising_input_tokens[i])).long()], 0)
    denoising_input_tokens = th.stack(denoising_input_tokens)

    denoising_output_tokens = [batch[i]["denoising_output_tokens"] for i in range(bs)]
    max_denoising_output_len = max(len(x) for x in denoising_output_tokens)
    for i in range(bs):
        if len(denoising_output_tokens[i]) < max_denoising_output_len:
            denoising_output_tokens[i] = th.cat([denoising_output_tokens[i], th.zeros(max_denoising_output_len - len(denoising_output_tokens[i])).long()], 0)
    denoising_output_tokens = th.stack(denoising_output_tokens)

    return {
        "video_id": video_id,
        "duration": duration,
        "video": video,
        "audio": audio,  # <-- [ADDED]
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "denoising_input_tokens": denoising_input_tokens,
        "denoising_output_tokens": denoising_output_tokens,
    }

# Builds the dense video captioning dataset for the specified split
def build_densevideocaptioning_dataset_audio(dataset_name, split, args, tokenizer):
    # Path to audio features directory for the right split.
    if dataset_name == "youcook":
        if split == "train":
            json_path = args.youcook_train_json_path
            audio_features_path = args.audio_features_train_path  
        elif split == "val":
            json_path = args.youcook_val_json_path
            audio_features_path = args.audio_features_val_path  
        else:
            raise NotImplementedError
        features_path = args.youcook_features_path
        subtitles_path = args.youcook_subtitles_path
    else:
        raise NotImplementedError

    return DenseVideoCaptioning_Dataset_Audio(
        json_path=json_path,
        features_path=features_path,
        audio_features_path=audio_features_path,  
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        tokenizer=tokenizer,
        subtitles_path=subtitles_path,
        num_bins=args.num_bins,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        noise_density=args.noise_density,
        mean_noise_span_length=args.mean_noise_span_length,
    )
