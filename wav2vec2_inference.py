import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class Wave2Vec2Inference():
    def __init__(self,model_name):
        """ Initialize the model with its tokenizer and predictor objects. This also downloads the model being used."""
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) 
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def buffer_to_text(self,audio_buffer):
        """ This function takes an audio data (audio chunk) as an input and transcribes it using the wav2vec2 model used in our case.
            It was invoked in live_asr.py to transcribe the audio chunks. 
        """
        if(len(audio_buffer)==0):
            return ""
        
        # The lines 18 to 25 all together do just one simple task- 'Transcription'.
        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)        
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()
