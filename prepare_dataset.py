import re
import json
import os
import html
import librosa
import numpy as np
from datetime import timedelta
from bs4 import BeautifulSoup

def parse_timestamp(timestamp):
    """Convert timestamp string (e.g., '00:00:00,000 --> 00:00:03,839') to start and end seconds."""
    try:
        match = re.match(r'.*?(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', timestamp)
        if not match:
            raise ValueError(f"Invalid timestamp format: {timestamp}")
        
        start_time, end_time = match.groups()
        
        start_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', start_time)
        start_hours, start_minutes, start_seconds, start_ms = map(int, start_match.groups())
        start_seconds = start_hours * 3600 + start_minutes * 60 + start_seconds + start_ms / 1000.0
        
        end_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', end_time)
        end_hours, end_minutes, end_seconds, end_ms = map(int, end_match.groups())
        end_seconds = end_hours * 3600 + end_minutes * 60 + end_seconds + end_ms / 1000.0
        
        return start_seconds, end_seconds
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {timestamp} - {str(e)}")

def extract_transcription_from_html(html_file):
    """Extract text and timestamps from HTML file using BeautifulSoup."""
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    segments = soup.select('div.segment')
    
    transcription = []
    for segment in segments:
        timestamp_span = segment.select_one('span.timestamp')
        if not timestamp_span:
            continue
        timestamp_str = timestamp_span.get_text(strip=True)
        try:
            start_seconds, end_seconds = parse_timestamp(timestamp_str)
        except ValueError:
            continue
        
        text_span = segment.select_one('span.text')
        if not text_span:
            continue
        text = html.unescape(text_span.get_text(strip=False, separator=' ')).strip()
        text = re.sub(r'\s+', ' ', text)
        
        if text and text != '.':
            transcription.append({
                'start': start_seconds,
                'end': end_seconds,
                'text': text,
                'timestamp_str': timestamp_str.split('·')[-1].strip()
            })
    
    return transcription

def segment_transcription(transcription, max_duration=25.0):
    """Segment transcription into batches of less than max_duration seconds."""
    segments = []
    current_segment = {'start': None, 'end': None, 'text': [], 'first_timestamp': None, 'last_timestamp': None}
    
    for entry in transcription:
        # If current segment is empty, initialize it
        if current_segment['start'] is None:
            current_segment['start'] = entry['start']
            current_segment['first_timestamp'] = entry['timestamp_str']
        
        # Check if adding this entry would exceed max_duration
        potential_end = entry['end']
        potential_duration = potential_end - current_segment['start']
        
        if potential_duration > max_duration and current_segment['text']:
            # Finalize current segment
            segments.append({
                'start': current_segment['start'],
                'end': current_segment['end'],
                'text': ' '.join(current_segment['text']),
                'timestamp_str': f"{current_segment['first_timestamp'].split(' --> ')[0]} --> {current_segment['last_timestamp'].split(' --> ')[1]}"
            })
            # Start new segment with current entry
            current_segment = {
                'start': entry['start'],
                'end': None,
                'text': [],
                'first_timestamp': entry['timestamp_str'],
                'last_timestamp': None
            }
        
        # Add current entry to segment
        current_segment['text'].append(entry['text'])
        current_segment['end'] = entry['end']
        current_segment['last_timestamp'] = entry['timestamp_str']
    
    # Add final segment if it exists
    if current_segment['text']:
        segments.append({
            'start': current_segment['start'],
            'end': current_segment['end'],
            'text': ' '.join(current_segment['text']),
            'timestamp_str': f"{current_segment['first_timestamp'].split(' --> ')[0]} --> {current_segment['last_timestamp'].split(' --> ')[1]}"
        })
    
    return segments

def load_audio_array(audio_file, sampling_rate=16000):
    """Load audio file and return array and sampling rate."""
    audio, sr = librosa.load(audio_file, sr=sampling_rate)
    return audio.tolist(), sr

def save_segments(segments, audio_file, output_dir, output_file_name, sampling_rate=16000):
    """Save segments to JSONL file for Whisper fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)
    jsonl_file = os.path.join(output_dir, output_file_name)
    
    audio_array, sr = load_audio_array(audio_file, sampling_rate)
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_array = audio_array[start_sample:end_sample]
            
            entry = {
                'sentence': segment['text'],
                'timestamps': {
                    'start': segment['start'],
                    'end': segment['end'],
                    'timestamp_str': segment['timestamp_str']
                },                
                'audio': {
                    'path': audio_file,
                    'sampling_rate': sr,
                    'array': segment_array
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return jsonl_file

def main():
    input_dir = "files"
    output_dir = "whisper_Dataset"
    max_duration = 25.0
    sampling_rate = 16000
    
    # Group files by their base name (without extension)
    files_by_base_name = {}
    for file_name in os.listdir(input_dir):
        base_name, ext = os.path.splitext(file_name)
        if base_name.endswith('_transcript'):
            base_name = base_name[:-11]
            
        if ext in ['.html', '.mp3']:
            if base_name not in files_by_base_name:
                files_by_base_name[base_name] = {}
            files_by_base_name[base_name][ext] = os.path.join(input_dir, file_name)
    
    if not files_by_base_name:
        raise FileNotFoundError("No HTML or MP3 files found in the input directory.")
    print("--"*50)
    print(f"Found {len(files_by_base_name)} pairs of HTML and MP3 files.")
    print("--"*50)
    print("\n\n")

    for base_name, files in files_by_base_name.items():
        html_file = files.get('.html')
        audio_file = files.get('.mp3')
    
        if not html_file or not audio_file:
            
            print("**"*50)
            print(f"Skipping {base_name}: Missing either HTML or MP3 file.")
            print("**"*50)
            print("\n\n")
            continue
        
        print("%%"*50)
        print(f"Processing {base_name}...")
        print("%%"*50)
        print("\n\n")
        transcription = extract_transcription_from_html(html_file)
        segments = segment_transcription(transcription, max_duration)
    
        # Create a unique output file name for each pair
        output_file_name = f"{base_name}_segments.jsonl"
    
        jsonl_file = save_segments(segments, audio_file, output_dir, output_file_name, sampling_rate)
    
        print(f"Generated {len(segments)} segments for {base_name} in {jsonl_file}")

if __name__ == "__main__":
    main()