from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from torchaudio.transforms import Resample
import os

checkpoint_path = "./whisper-large-finetuned/final"

print("Loading processor...")
processor = WhisperProcessor.from_pretrained('./whisper-large-finetuned')
print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

# ✅ Disable forced decoder ids from BOTH config objects
model.config.forced_decoder_ids = None
model.generation_config.forced_decoder_ids = None
model.eval()

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# Load and resample audio
def load_audio(path, target_sr=16000):
    print(f"Loading audio from: {path}")
    speech_array, sr = torchaudio.load(path)
    if speech_array.shape[0] > 1:
        print("Converting stereo to mono...")
        speech_array = speech_array.mean(dim=0, keepdim=True)
    if sr != target_sr:
        print(f"Resampling from {sr} to {target_sr}...")
        resampler = Resample(orig_freq=sr, new_freq=target_sr)
        speech_array = resampler(speech_array)
    return speech_array.squeeze(0), target_sr

# Split audio into chunks
def chunk_audio(audio_tensor, chunk_size=30*16000):  # 30 seconds
    print("Chunking audio...")

    chunks = []
    for i in range(0, len(audio_tensor), chunk_size):
        chunks.append(audio_tensor[i:i+chunk_size])

    print(f"Total chunks: {len(chunks)}")
    return chunks

# Transcribe long audio
def transcribe_audio_in_chunks(audio_path):
    waveform, sr = load_audio(audio_path)
    chunks = chunk_audio(waveform)

    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"\nTranscribing chunk {i+1}/{len(chunks)}...")
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcript += transcription.strip() + " "

    return full_transcript.strip()

def transcribe_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".mp3"):
            audio_path = os.path.join(input_dir, filename)
            print(f"\n\nStarting full transcription for {audio_path}...")
            transcription = transcribe_audio_in_chunks(audio_path)

            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{base_name}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription)

            print(f"\n\nSaved transcription to {output_file}")

INPUT_DIR = "inbox"
OUTPUT_DIR = "Transcriptions"

if __name__ == "__main__":
    transcribe_folder(INPUT_DIR, OUTPUT_DIR)
