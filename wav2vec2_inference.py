import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class Wave2Vec2Inference():
    def __init__(self,model_name):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) 
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def buffer_to_text(self,audio_buffer):
        if(len(audio_buffer)==0):
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)        
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()
