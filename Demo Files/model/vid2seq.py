# File: vid2seq.py
# Original Author(s): VidChapters Team (https://github.com/antoyang/VidChapters)
# Modified by: Daniel Vousden
#
# Description:
# This file extends the original VidChapters Vid2Seq model to incorporate audio as an additional input modality,
# in addition to the existing video and ASR (speech) inputs. Audio features (e.g., from Wav2Vec) are integrated
# into the encoder's input sequence, with corresponding attention masks, allowing the transformer to jointly
# reason over all three modalities.
#
# Key Modifications:
# - Added 'use_audio' flag to enable audio modality in the model.
# - Added 'audio' argument to the `forward` and `generate` methods.
# - Constructed attention masks for audio features and appended them to the encoder input.
# - Fused audio embeddings with video and ASR token embeddings using concatenation.
#
# These changes enable the model to capture additional contextual cues from raw audio signals, enhancing
# the temporal and semantic richness of generated captions.
#
# License: MIT


import torch
import torch.nn as nn
from .modeling_t5 import T5ForConditionalGeneration
from .vit import VisionTransformer
from transformers import T5Tokenizer
from transformers.modeling_outputs import (
    BaseModelOutput,
)

def _get_tokenizer(tokenizer_path, num_bins=0):
    if 't5' in tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if num_bins:
            new_tokens = ["<time=" + str(i) + ">" for i in range(num_bins)]
            tokenizer.add_tokens(list(new_tokens))
    else:
        raise NotImplementedError(tokenizer_path)
    return tokenizer

class Vid2Seq(torch.nn.Module):
    def __init__(self,
                 t5_path,
                 num_features=100,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=2048,
                 vis_drop=0.,
                 tokenizer=None,
                 enc_drop=0.,
                 dec_drop=0.1,
                 use_speech=True,
                 use_video=True,
                 num_bins=100,
                 label_smoothing=0.1):
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(encoder_dropout=enc_drop, decoder_dropout=dec_drop, label_smoothing=label_smoothing,
                                                                   pretrained_model_name_or_path=t5_path, local_files_only=True, is_gated_act="v1_1" in t5_path)
        self.t5_model.resize_token_embeddings(len(tokenizer) - num_bins)
        self.t5_model.resize_token_embeddings(len(tokenizer))
        self.visual_encoder = VisionTransformer(num_features=num_features,
                                                embed_dim=embed_dim,
                                                depth=depth,
                                                num_heads=heads,
                                                mlp_dim=mlp_dim,
                                                qkv_bias=True,
                                                qk_scale=None,
                                                drop_rate=vis_drop,
                                                attn_drop_rate=vis_drop,
                                                norm_layer=nn.LayerNorm)
        self.t5_tokenizer = tokenizer
        self.use_speech = use_speech
        self.use_video = use_video
        self.use_audio = True  # toggle for audio input
        self.proj_v2t = None
        if self.t5_model.model_dim != 768:
            self.proj_v2t = nn.Linear(768, self.t5_model.model_dim)
     # added audio parameter for forward passes.
    def forward(self, video, input_tokenized, output_tokenized, audio=None): 
        if self.use_video:
            if isinstance(video, dict):
                video, atts_vis = video["video"], video["atts_vis"]
            else:
                video = self.visual_encoder(video)
                if self.proj_v2t is not None:
                    video = self.proj_v2t(video)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)
            video_dict = {"video": video, "atts_vis": atts_vis}
        else:
            video_dict = None

        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        if self.use_audio and audio is not None:
            # attention mask for audio
            atts_audio = torch.ones(audio.size()[:-1], dtype=torch.long).to(audio.device) 
        else:
            atts_audio = None

        if self.use_video and self.use_speech:
            hidden_states = [video, encoded.last_hidden_state]
            attention_masks = [atts_vis, input_tokenized['attention_mask']]
        elif self.use_video:
            hidden_states = [video]
            attention_masks = [atts_vis]
        elif self.use_speech:
            hidden_states = [encoded.last_hidden_state]
            attention_masks = [input_tokenized['attention_mask']]
        else:
            hidden_states, attention_masks = [], []

        if self.use_audio and audio is not None:
            # audio to hidden states and attention mask
            hidden_states.append(audio)  
            attention_masks.append(atts_audio)  

        final_hidden_state = torch.cat(hidden_states, dim=1)
        encoder_atts = torch.cat(attention_masks, dim=1)

        targets = output_tokenized['input_ids'].masked_fill(
            output_tokenized['input_ids'] == self.t5_tokenizer.pad_token_id, -100
        )
        outputs = self.t5_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=final_hidden_state), # fused multimodal embeddings
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokenized['attention_mask'],
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}, video_dict

    @torch.no_grad()
    def generate(
            self,
            video,
            input_tokenized,
            audio=None,  # added audio parameter
            use_nucleus_sampling=False,
            num_beams=4,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        if self.use_video:
            video = self.visual_encoder(video)
            if self.proj_v2t is not None:
                video = self.proj_v2t(video)
            atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

        if self.use_audio and audio is not None:
            # attention mask for audio
            atts_audio = torch.ones(audio.size()[:-1], dtype=torch.long).to(audio.device) 

        hidden_states, attention_masks = [], []
        if self.use_video:
            hidden_states.append(video)
            attention_masks.append(atts_vis)

        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )
            hidden_states.append(encoded.last_hidden_state)
            attention_masks.append(input_tokenized['attention_mask'])

        if self.use_audio and audio is not None:
            # Added: audio to hidden states and attentio mask
            hidden_states.append(audio)  
            attention_masks.append(atts_audio) 

        final_hidden_state = torch.cat(hidden_states, dim=1)
        encoder_atts = torch.cat(attention_masks, dim=1)

        outputs = self.t5_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=final_hidden_state),  # fused multimodal embeddings
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text
