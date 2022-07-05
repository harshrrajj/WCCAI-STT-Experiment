#!/usr/local/bin/python3
import webrtcvad
import pyaudio
from live_asr import LiveWav2Vec2


################################

englishModel = "facebook/wav2vec2-large-960h-lv60-self"  # This model(from the huggingface hub) works better than "wav2vec2-base-960h" and many others.
# The only dialects involved: (US, EU, CN)

asr = LiveWav2Vec2(englishModel,device_name="default")   # calls the LiveWav2Vec2 class used in live_asr.py
# live_asr.py or LiveWav2Vec2 class of that program is the heart of this project. It's been invoked to perfrom the live transcription as shown below.

################################

# Refer to LiveWav2Vec2 class of live_asr.py file.
asr.start()

try:
    while True:
        text,sample_length,inference_time = asr.get_last_text()                        
        print(f"{sample_length:.3f}s"
        +f"\t{inference_time:.3f}s"
        +f"\t{text}")
        
except KeyboardInterrupt:   
    asr.stop()
