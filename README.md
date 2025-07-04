# Whisper Model Fine-Tuning Guide

This repository provides a complete guide and scripts for fine-tuning the Whisper model for custom audio transcription tasks.

## Project Structure

```
.
├── .gitignore
├── Requirements.txt
├── Transcriptions/             # Stores generated transcriptions
├── files/                      # Contains audio files and their initial HTML transcripts
├── finetune_whisper.py         # Script for fine-tuning the Whisper model
├── generate-transcript-using-whisper.py # Script for transcribing audio using a fine-tuned model
├── inbox/                      # Input directory for audio files to be transcribed
├── prepare_dataset.py          # Script for preparing the dataset for fine-tuning
└── whisper_Dataset/            # Stores prepared dataset in JSONL format
```

## Scripts Overview

This project includes three main Python scripts to guide you through the process of preparing data, fine-tuning the Whisper model, and generating transcriptions.

### 1. `prepare_dataset.py`

This script is responsible for preparing your audio and transcription data into a format suitable for Whisper model fine-tuning. It handles parsing HTML transcription files, converting audio to the required format (16000Hz, max 30 seconds segments), and saving the processed data as a JSONL file.

**Usage:**

1.  Place your audio files (e.g., `.mp3`) in the `files/` directory.
2.  Ensure you have corresponding HTML transcription files in the `files/` directory (e.g., `gra-on-mishlei---50_transcript.html` for `gra-on-mishlei---50.mp3`).
3.  Run the script:
    ```bash
    python prepare_dataset.py
    ```

**Output:**

The script will process the data and save it in the `whisper_Dataset/` directory as a JSONL file (e.g., `gra-on-mishlei---50_segments.jsonl`). This JSONL file will contain segments of audio and their corresponding transcriptions, ready for model training.

### 2. `finetune_whisper.py`

This script handles the core fine-tuning process of the Whisper model. It loads the prepared dataset from the `whisper_Dataset/` directory, fine-tunes the model, and saves the fine-tuned model.

**Usage:**

1.  Ensure you have prepared your dataset using `prepare_dataset.py` and the `whisper_Dataset/` directory contains the necessary JSONL files.
2.  Run the script:
    ```bash
    python finetune_whisper.py
    ```

**Output:**

Upon successful completion, the fine-tuned Whisper model will be saved in a new directory named `whisper-large-finetuned/` in the project root.

### 3. `generate-transcript-using-whisper.py`

This script allows you to use a fine-tuned Whisper model to transcribe new audio files. It loads the saved model, takes an audio file from the `inbox/` directory, transcribes it, and saves the transcription.

**Usage:**

1.  Place the audio file you wish to transcribe in the `inbox/` directory (e.g., `shavuos-7---after-being-up-all-night-can-you-continue-learning-after-alos-hashachar-do-you-need-to-wash-netilas-yadayim-at-alos.mp3`).
2.  Ensure you have a fine-tuned model available in the `whisper-large-finetuned/` directory (generated by `finetune_whisper.py`).
3.  Run the script:
    ```bash
    python generate-transcript-using-whisper.py
    ```

**Output:**

The generated transcription will be saved as a text file in the `Transcriptions/` directory.