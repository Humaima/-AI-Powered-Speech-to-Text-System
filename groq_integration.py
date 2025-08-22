from groq import Groq
from config import config

class GroqPostProcessor:
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL
        
    def correct_transcription(self, text):
        """Use LLM to correct and improve transcription"""
        prompt = f"""
        Please correct and improve the following speech transcription. 
        Fix any grammatical errors, punctuation, and make it more readable.
        Return only the corrected text without any additional commentary.
        
        Transcription: {text}
        
        Corrected transcription:
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that corrects speech transcriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq API error: {e}")
            return text  # Return original text if API fails
    
    def summarize_text(self, text):
        """Optional: Summarize long transcriptions"""
        prompt = f"""
        Please provide a concise summary of the following text:
        
        {text}
        
        Summary:
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq API error: {e}")
            return "Summary not available"