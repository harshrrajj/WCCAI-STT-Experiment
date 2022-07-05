#!/usr/local/bin/python3
import webrtcvad
import pyaudio
from live_asr import LiveWav2Vec2


################################

englishModel = "facebook/wav2vec2-large-960h-lv60-self"  # The only dialects involved: (US, EU, CN)
#This model(from the huggingface hub) often works better thatn wav2vec2-base-960h

asr = LiveWav2Vec2(englishModel,device_name="default")   # This downloads the model and instantiates it as well as specifies the device to be used for microphone.

################################


asr.start()

try:
    while True:
        text,sample_length,inference_time = asr.get_last_text()                        
        print(f"{sample_length:.3f}s"
        +f"\t{inference_time:.3f}s"
        +f"\t{text}")
        
except KeyboardInterrupt:   
    asr.stop()
