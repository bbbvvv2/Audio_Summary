#!/usr/bin/env python3
"""
Local Audio Summarization Pipeline

Dependencies:
    pip install git+https://github.com/openai/whisper.git
    pip install transformers torch

Usage:
    python local_audio_summary.py input_audio.mp3 --model small --min_length 50 --max_length 200

Outputs:
    transcript.txt
    summary.txt
"""

import whisper
from transformers import pipeline
import argparse

def transcribe(audio_path, model_name="small"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    transcript_text = result["text"]
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript_text)
    return transcript_text


def summarize(text, min_length=50, max_length=200):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, min_length=min_length, max_length=max_length)
    summary_text = summary[0]["summary_text"]
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    return summary_text


def main():
    parser = argparse.ArgumentParser(description="Local Audio Summarization Pipeline")
    parser.add_argument("audio_file", help="Path to audio file (mp3, wav, etc.)")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--min_length", type=int, default=50, help="Min summary length")
    parser.add_argument("--max_length", type=int, default=200, help="Max summary length")
    args = parser.parse_args()
    
    print("Transcribing...")
    transcript = transcribe(args.audio_file, args.model)
    print("Transcription saved to transcript.txt")
    
    print("Summarizing...")
    summary = summarize(transcript, args.min_length, args.max_length)
    print("Summary saved to summary.txt")
    
    print("Done.")

if __name__ == "__main__":
    main()
