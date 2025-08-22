import os

class Config:
    # Audio Settings
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 30 # Seconds per Chunk
    MAX_AUDIO_LENGTH = 300 # maximum audio length in seconds

    # Model settings
    SPEECH_MODEL_NAME = "facebook/wav2vec2-base-960h"  # Alternative to Whisper
    # Other options: "facebook/wav2vec2-large-960h-lv60-self", "patrickvonplaten/wav2vec2-large-960h-lv60-self"

    # Groq API settings
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    GROQ_MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768", "gemma-7b-it"

    # File paths
    TEMP_AUDIO_DIR = "temp_audio"

config = Config()