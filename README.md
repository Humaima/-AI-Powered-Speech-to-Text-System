# 🎤 Speech-to-Text System
AI-Powered Audio Transcription with Modern Architecture

![pic](https://github.com/user-attachments/assets/3663d8b1-d2d8-43dc-9677-1839fca227fc)

---

## 🚀 Overview
This project provides a **robust, scalable speech-to-text pipeline** that combines:
- **Transformer-based speech recognition (Wav2Vec2)**
- **Audio preprocessing & chunking**
- **LLM-based transcription correction (Groq LLaMA-3.1)**
- **Interactive Streamlit frontend for file uploads & transcription download**

The system is modular, maintainable, and built with modern AI and software engineering practices.

---


---

## ⚙️ Features
- 🎙 **Audio Processing**: Normalization, resampling, mono conversion, chunking for long files  
- 🤖 **AI Recognition**: Hugging Face Wav2Vec2 model for transcription  
- 🧠 **LLM Enhancement**: Post-processing with Groq’s LLaMA-3.1 model  
- 🛠 **Pipeline Orchestration**: Modular, error-handled, configurable workflow  
- 💻 **Web Interface**: Upload audio files, track progress, view results, download transcription  

---

## 🔑 Key Components
1. **Configuration Management (`config.py`)**
   - Centralized constants & environment variables
   - Supports secure API key management  

2. **Audio Processing (`audio_processor.py`)**
   - Signal normalization & cleaning  
   - Chunking into smaller audio segments for efficiency  

3. **Speech Recognition (`speech_recognizer.py`)**
   - Wav2Vec2 transformer model with GPU acceleration  
   - Uses CTC (Connectionist Temporal Classification) for alignment  

4. **LLM Integration (`groq_integration.py`)**
   - API-based correction of transcriptions  
   - Improves grammar, readability, and punctuation  

5. **Pipeline (`main.py`)**
   - End-to-end orchestration of audio → text pipeline  
   - Handles errors gracefully and supports optional correction  

6. **Frontend (`app.py`)**
   - Built with **Streamlit**  
   - Upload audio (WAV, MP3, M4A, FLAC, OGG)  
   - Real-time progress tracking  
   - Download transcription as a text file  

---

## 🏗 Architecture
- **MVC Pattern**
  - Model → `audio_processor.py`, `speech_recognizer.py`
  - View → `app.py`
  - Controller → `main.py`
- **Dependency Injection**: Modular and testable
- **Robust Error Handling**: Fail-safe mechanisms
- **Resource Management**: Temporary storage & cleanup
- **Scalability**: Easy to extend with other models/APIs

---

## 📦 Installation
```bash
# Clone repo
git clone https://github.com/your-username/speech-to-text-system.git
cd speech-to-text-system
```

# Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```
# Install dependencies
```bash
pip install -r requirements.txt
```
## 🔑 Environment Variables

Create a .env file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

## ▶️ Usage

Run Streamlit App
```bash
streamlit run app.py
```
- Upload an audio file
- Wait for processing
- View & download transcription

Run from CLI
```bash
python main.py --file your_audio.wav
```
## ✅ Example

Input: sample_audio.wav

Output:
```bash
"Hello, everyone. Thank you for joining today’s meeting..."
```

## 🌟 Benefits

- Modular & scalable design
- State-of-the-art AI integration
- Error-resilient pipeline
- User-friendly interface

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributing

Pull requests are welcome!
