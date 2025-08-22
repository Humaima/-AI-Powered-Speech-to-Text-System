import streamlit as st
import os
import time
import tempfile
from audio_processor import AudioProcessor
from speech_recognizer import SpeechRecognizer
from groq_integration import GroqPostProcessor
from config import config

# Set page configuration
st.set_page_config(
    page_title="Speech-to-Text Pipeline",
    page_icon="🎤",
    layout="wide"
)

class SpeechToTextPipeline:
    def __init__(self):
        st.session_state.setdefault("initialized", False)
        
        if not st.session_state.initialized:
            with st.spinner("Initializing audio processor..."):
                self.audio_processor = AudioProcessor()
            
            with st.spinner("Initializing speech recognizer..."):
                self.speech_recognizer = SpeechRecognizer()
            
            with st.spinner("Initializing Groq processor..."):
                self.groq_processor = GroqPostProcessor()
            
            st.session_state.initialized = True
            st.session_state.pipeline = self
        else:
            self.audio_processor = st.session_state.pipeline.audio_processor
            self.speech_recognizer = st.session_state.pipeline.speech_recognizer
            self.groq_processor = st.session_state.pipeline.groq_processor
        
    def process_audio_file(self, audio_path, use_groq_correction=True):
        """Process audio file through the entire pipeline"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load and preprocess audio
            status_text.text("Loading audio...")
            audio_array = self.audio_processor.load_audio(audio_path)
            progress_bar.progress(20)
            
            status_text.text(f"Audio length: {len(audio_array)/config.SAMPLE_RATE:.2f} seconds")
            
            status_text.text("Preprocessing audio...")
            audio_array = self.audio_processor.preprocess_audio(audio_array)
            progress_bar.progress(40)
            
            # Chunk audio if too long
            if len(audio_array) > config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE:
                status_text.text("Audio is too long, chunking...")
                chunks = self.audio_processor.chunk_audio(audio_array)
                status_text.text(f"Split into {len(chunks)} chunks")
                status_text.text("Transcribing chunks...")
                transcription = self.speech_recognizer.transcribe_chunks(chunks)
            else:
                status_text.text("Transcribing audio...")
                transcription = self.speech_recognizer.transcribe_audio(audio_array)
            
            progress_bar.progress(80)
            
            # Post-process with Groq
            if use_groq_correction and transcription.strip():
                status_text.text("Post-processing with Groq...")
                corrected_transcription = self.groq_processor.correct_transcription(transcription)
                result = corrected_transcription
            else:
                result = transcription
                
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            return result
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error in processing pipeline: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return f"Error: {str(e)}"

def main():
    st.title("🎤 Speech-to-Text Pipeline")
    st.markdown("Upload an audio file to transcribe it using transformer models with optional Groq post-processing.")
    
    # Initialize pipeline
    pipeline = SpeechToTextPipeline()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg']
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        use_groq = st.checkbox("Use Groq post-processing", value=True)
    with col2:
        if st.button("Clear Cache", help="Clear all cached data and reinitialize models"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file)
        
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        st.write(file_details)
        
        if st.button("Transcribe Audio"):
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            try:
                # Process audio
                start_time = time.time()
                result = pipeline.process_audio_file(audio_path, use_groq)
                processing_time = time.time() - start_time
                
                # Display results
                st.subheader("Transcription Result")
                st.text_area("", result, height=200)
                
                st.info(f"Processing time: {processing_time:.2f} seconds")
                
                # Download button
                st.download_button(
                    label="Download Transcription",
                    data=result,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcription.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                # Clean up temporary file
                os.unlink(audio_path)

if __name__ == "__main__":
    main()