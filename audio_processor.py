import librosa
import numpy as np
from pydub import AudioSegment
import os
import soundfile as sf
from config import config
import warnings

class AudioProcessor:
    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        self._setup_ffmpeg()
        
    def _setup_ffmpeg(self):
        """Setup FFmpeg path for pydub"""
        # Common FFmpeg installation paths
        possible_ffmpeg_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\tools\ffmpeg\bin\ffmpeg.exe",
            os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
        ]
        
        # Check if ffmpeg is in PATH
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            print("FFmpeg found in PATH")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try to find ffmpeg in common locations
        for ffmpeg_path in possible_ffmpeg_paths:
            if os.path.exists(ffmpeg_path):
                AudioSegment.converter = ffmpeg_path
                print(f"FFmpeg found at: {ffmpeg_path}")
                return
        
        print("Warning: FFmpeg not found. Some audio formats may not work properly.")
        
    def load_audio(self, audio_path):
        """Load audio file and convert to required format"""
        try:
            # Use librosa for all formats to avoid ffmpeg dependency
            print(f"Loading audio with librosa: {audio_path}")
            audio_array, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio_array
            
        except Exception as e:
            print(f"Librosa failed: {e}. Trying alternative methods...")
            return self._load_audio_alternative(audio_path)
    
    def _load_audio_alternative(self, audio_path):
        """Alternative audio loading method"""
        try:
            # Try soundfile first
            audio_array, sr = sf.read(audio_path)
            if sr != self.sample_rate:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=0)
            return audio_array
        except Exception as e:
            print(f"Soundfile failed: {e}. Trying pydub with warning suppression...")
            return self._load_audio_with_pydub(audio_path)
    
    def _load_audio_with_pydub(self, audio_path):
        """Load audio with pydub (with suppressed warnings)"""
        try:
            # Suppress warnings temporarily
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
                audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
                audio_array /= 2**15  # Normalize to [-1, 1] for 16-bit audio
                return audio_array
        except Exception as e:
            raise Exception(f"All audio loading methods failed: {str(e)}")
    
    def preprocess_audio(self, audio_array):
        """Preprocess audio for model input - THIS WAS MISSING"""
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=0)
            
        # Normalize
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val
        
        # Remove silence (optional)
        audio_array = self.remove_silence(audio_array)
        
        return audio_array
    
    def remove_silence(self, audio_array, threshold=0.01):
        """Remove leading and trailing silence"""
        # Find non-silent parts
        non_silent = np.where(np.abs(audio_array) > threshold)[0]
        
        if len(non_silent) > 0:
            start = non_silent[0]
            end = non_silent[-1]
            return audio_array[start:end]
        
        return audio_array
    
    def chunk_audio(self, audio_array, chunk_duration=config.CHUNK_DURATION):
        """Split audio into chunks for processing"""
        chunk_size = int(chunk_duration * self.sample_rate)
        chunks = []
        
        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i:i + chunk_size]
            if len(chunk) > self.sample_rate:  # At least 1 second
                chunks.append(chunk)
                
        return chunks
    
    def convert_to_tensor(self, audio_array):
        """Convert numpy array to torch tensor"""
        import torch
        return torch.from_numpy(audio_array).float()
    
    def save_audio(self, audio_array, output_path):
        """Save audio array to file"""
        sf.write(output_path, audio_array, self.sample_rate)