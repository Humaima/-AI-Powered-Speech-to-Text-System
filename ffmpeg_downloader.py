import os
import zipfile
import requests
import tempfile
import shutil
from tqdm import tqdm
from pydub import AudioSegment

def download_ffmpeg():
    """Download and extract FFmpeg automatically"""
    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    ffmpeg_bin = os.path.join(ffmpeg_dir, "bin", "ffmpeg.exe")
    
    # Check if already downloaded
    if os.path.exists(ffmpeg_bin):
        print("FFmpeg already downloaded")
        return ffmpeg_bin
    
    print("Downloading FFmpeg...")
    
    # URL for FFmpeg Windows build
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    try:
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        temp_zip = os.path.join(tempfile.gettempdir(), "ffmpeg.zip")
        
        # Download with progress bar
        total_size = int(response.headers.get('content-length', 0))
        with open(temp_zip, 'wb') as f, tqdm(
            desc="Downloading FFmpeg",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        # Extract the zip file
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        # Clean up
        os.remove(temp_zip)
        
        print(f"FFmpeg downloaded to: {ffmpeg_dir}")
        return ffmpeg_bin
        
    except Exception as e:
        print(f"Failed to download FFmpeg: {e}")
        return None

# Add this to your audio_processor.py __init__ method
def _setup_ffmpeg(self):
    """Setup FFmpeg path for pydub"""
    # Try to download ffmpeg if not found
    ffmpeg_bin = download_ffmpeg()
    if ffmpeg_bin and os.path.exists(ffmpeg_bin):
        AudioSegment.converter = ffmpeg_bin
        print(f"Using FFmpeg at: {ffmpeg_bin}")
    else:
        print("Warning: FFmpeg not available. Using librosa for audio processing.")