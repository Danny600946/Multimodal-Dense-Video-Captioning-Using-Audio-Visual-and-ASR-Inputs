# Multimodal-Dense-Video-Captioning-Using-Audio-Visual-and-ASR-Inputs
This dissertation explores enhancing dense video captioning by integrating audio into the Vid2Seq pipeline. By leveraging multimodal data, the project aims to improve caption accuracy and context, investigating the potential of audio-visual fusion for richer, more informative video descriptions. 

Details For Model and Data Downloads:

Fine Tuning and Evaluation (Fine Tuning and Evaluation Folder):

Requires the t5-base model to be downloaded to the t5-base model folder. All files from this code repo link should be placed in the t5-base folder (Can be found at https://huggingface.co/google-t5/t5-base/tree/main)

Requires the vid2seq checkpoints to be downloaded from the VidChapters git and be placed in the vid2seq folder. Found under 'Model Checkpoints' and called 'HowTo100M + VidChapters-7M + YouCook2' (Found at https://github.com/antoyang/VidChapters)

Requires the vit youCook2 vit features .pth to be downloaded and placed in the YouCook2 folder. Found under 'Data Downloading' (Found at https://github.com/antoyang/VidChapters) (Or on this drive https://drive.google.com/drive/folders/1hTDCIZU_TOB0a5jvRhY98lDChe93Tcqs)

Requires the youCook2 asr.pkl to be downloaded and placed in the YouCook2 folder. Found under 'Data Downloading' (Found at https://github.com/antoyang/VidChapters) (Or on this drive https://drive.google.com/drive/folders/1hTDCIZU_TOB0a5jvRhY98lDChe93Tcqs)

Once these requirements are satisfied the reproducibility section of my dissertation report can be followed.

It should be noted, due to the upload limitations the download and extraction files for the audio features will have to be run as detailed in the reproducibility section of my dissertation report.

Demo (Demo Folder):

Requires the t5 model to be downloaded to TRANSFORMER_CAHCE/t5-base. All files form this code repo link should be place in the t5-base folder (Can be found at https://huggingface.co/google-t5/t5-base/tree/main)

Requires Whisper-Large-V2 to be downloaded into the MODEL_DIR folder. This should happen automatically when running demo_asr.py (Refer to reproducibility section in Dissertation report)

Requires Vit-L-14 to be downloaded into the MODEL_DIR folder. This should happen automatically when running demo_vidseq.py (Refer to reproducibility section in Dissertation report)

Note: Since this is the demo, you can choose any Vid2Seq checkpoints. However, they must be placed in MODEL_DIR/vid2seq (checkpoints are required). Pretrained checkpoints can be downloaded from VidChapters (https://github.com/antoyang/VidChapters)

Once these requirements are satisfied the reproducibility section of my dissertation report can be followed.

The dissertation report outlines prior research, implementation strategy, usage instructions, and performance metrics.
