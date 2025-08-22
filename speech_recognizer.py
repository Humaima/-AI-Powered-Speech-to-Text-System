from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import numpy as np
from config import config

class SpeechRecognizer:
    def __init__(self, model_name=config.SPEECH_MODEL_NAME):
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
    def transcribe_audio(self, audio_array):
        """Transcribe audio using Wav2Vec2 model"""
        try:
            # Ensure audio is numpy array
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.numpy()
            
            # Check if audio is too short
            if len(audio_array) < config.SAMPLE_RATE:  # Less than 1 second
                audio_array = np.pad(audio_array, (0, config.SAMPLE_RATE - len(audio_array)))
            
            # Preprocess audio
            inputs = self.processor(
                audio_array, 
                sampling_rate=config.SAMPLE_RATE, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(input_values).logits
                
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)
            
            return transcription[0] if isinstance(transcription, list) else transcription
            
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")
    
    def transcribe_chunks(self, audio_chunks):
        """Transcribe multiple audio chunks"""
        transcriptions = []
        
        for i, chunk in enumerate(audio_chunks):
            print(f"Processing chunk {i+1}/{len(audio_chunks)}...")
            try:
                transcription = self.transcribe_audio(chunk)
                transcriptions.append(transcription)
                print(f"Chunk {i+1}: {transcription[:100]}...")
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                transcriptions.append("")
            
        return " ".join(transcriptions)