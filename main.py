import argparse
import os
import sys
import time
from audio_processor import AudioProcessor
from speech_recognizer import SpeechRecognizer
from groq_integration import GroqPostProcessor
from config import config

class SpeechToTextPipeline:
    def __init__(self):
        print("Initializing audio processor...")
        self.audio_processor = AudioProcessor()
        print("Initializing speech recognizer...")
        self.speech_recognizer = SpeechRecognizer()
        print("Initializing Groq processor...")
        self.groq_processor = GroqPostProcessor()
        
    def process_audio_file(self, audio_path, use_groq_correction=True):
        """Process audio file through the entire pipeline"""
        print(f"Processing audio file: {audio_path}")
        
        try:
            # Load and preprocess audio
            print("Loading audio...")
            audio_array = self.audio_processor.load_audio(audio_path)
            print(f"Audio length: {len(audio_array)/config.SAMPLE_RATE:.2f} seconds")
            
            print("Preprocessing audio...")
            audio_array = self.audio_processor.preprocess_audio(audio_array)
            
            # Chunk audio if too long
            if len(audio_array) > config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE:
                print("Audio is too long, chunking...")
                chunks = self.audio_processor.chunk_audio(audio_array)
                print(f"Split into {len(chunks)} chunks")
                transcription = self.speech_recognizer.transcribe_chunks(chunks)
            else:
                print("Transcribing audio...")
                transcription = self.speech_recognizer.transcribe_audio(audio_array)
        
            print(f"\nRaw transcription:\n{transcription}")
            
            # Post-process with Groq
            if use_groq_correction and transcription.strip():
                print("Post-processing with Groq...")
                corrected_transcription = self.groq_processor.correct_transcription(transcription)
                print(f"\nCorrected transcription:\n{corrected_transcription}")
                return corrected_transcription
            else:
                return transcription
                
        except Exception as e:
            print(f"Error in processing pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text with Transformer Models")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--no-groq", action="store_true", help="Disable Groq post-processing")
    parser.add_argument("--model", type=str, default="", help="Specific model to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        return
    
    # Initialize pipeline
    pipeline = SpeechToTextPipeline()
    
    try:
        start_time = time.time()
        
        # Process audio
        result = pipeline.process_audio_file(args.audio, not args.no_groq)
        
        processing_time = time.time() - start_time
        
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        print(result)
        print(f"\nProcessing time: {processing_time:.2f} seconds")
        
        # Save result to file
        output_file = f"{os.path.splitext(args.audio)[0]}_transcription.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        print(f"\nTranscription saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()