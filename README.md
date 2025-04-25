# Audio Summarizer

Python CLI using Whisper and Hugging Face Transformers.

## Setup

```bash
pip install -r requirements.txt
python local_audio_summary.py input_audio.mp3 \
  --model small --min_length 50 --max_length 200

git clone https://github.com/…/Audio-Summary.git
cd Audio-Summary
