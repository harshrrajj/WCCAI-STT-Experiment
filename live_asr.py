import pyaudio
import webrtcvad
from wav2vec2_inference import Wave2Vec2Inference
import numpy as np
import threading
import time
from sys import exit
from queue import  Queue


class LiveWav2Vec2():
    exit_event = threading.Event()    
    def __init__(self, model_name, device_name="default"):
        """ This function initializes the model and device names.
            Here, "facebook/wav2vec2-large-960h-lv60-self" will go as the model name. The device name is required only to use the device microphone
        """
        self.model_name = model_name
        self.device_name = device_name              

    def stop(self):
        """stop the automatic speech recognition (asr) process"""
        LiveWav2Vec2.exit_event.set()
        self.asr_input_queue.put("close")
        print("asr stopped")

    def start(self):
        """start the asr process"""
        self.asr_output_queue = Queue()     # This queue is used to store the transcriptions(output) to be printed on conversion from speech to text .
        self.asr_input_queue = Queue()      # This queue is used to store the audio chunks(input).
        self.asr_process = threading.Thread(target=LiveWav2Vec2.asr_process, args=(       # The threading used is quite generic, nothing much!
            self.model_name, self.asr_input_queue, self.asr_output_queue,))
        self.asr_process.start()
        time.sleep(5)  # start vad thread after asr model is loaded
        self.vad_process = threading.Thread(target=LiveWav2Vec2.vad_process, args=(
            self.device_name, self.asr_input_queue,))
        self.vad_process.start()

    def vad_process(device_name, asr_input_queue):
        """ The device name taken as an input is used to find a microphone that streams live audio input.
            This function perfroms audio streaming. It also initializes a webrtcvad object, which detects whether a given audio chunk is voiced or unvoiced.
            The chunking of audio stream is done based on certain PyAudio and webrtcvad constraints shown below. After execution of this function, we are left with
            an input queue (asr_input_queue) filled with several chunks of audio data.
        """
        vad = webrtcvad.Vad()   # webrtcvad object 
        vad.set_mode(1)         # general
        
        # pyaudio is instated to work with the device microphone
        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1    # single channel speech coming through microphone
        RATE = 16000    # The sampling rate must be 16k Hz when dealing with wav2vec2 models.
        # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
        FRAME_DURATION = 30
        CHUNK = int(RATE * FRAME_DURATION / 1000)  # Number of frames for FRAME_DURATION ms of duration. This is also called Chunk Size!

        microphones = LiveWav2Vec2.list_microphones(audio)
        selected_input_device_id = LiveWav2Vec2.get_input_device_id(
            device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,   # pass all the parameters' values from above. 
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''  # frames = 0 (The number of frames)      
        
        while True:   # This loop runs until we interrupt the kernel. It basically takes a continuous audio input from the microhpone.
            if LiveWav2Vec2.exit_event.is_set():
                break            
            frame = stream.read(CHUNK)    # read the audio-stream for chunk size of 'CHUNK'
            is_speech = vad.is_speech(frame, RATE)  # determine if the read chunk is silent or speeched.
            if is_speech:
                frames += frame    # add the read chunk to the frames variable if it is speeched.
            else:
                if len(frames) > 1:
                    # if the last chunk was silent, the whole transcription process will be shifted to a new line, storing the audio gathered so far (in the loop) in 
                    # the input queue. Note that the silence check interval is 30 ms as well i.e., if the speaker remains silent for atleast 30 ms, the transcription shifts to a new line.
                    asr_input_queue.put(frames)
                frames = b''  # frames is again initialized to 0 for restarting the whole process above.
        # end and terminate the audio streaming.
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def asr_process(model_name, in_queue, output_queue):
        """ This function takes the input queue filled from executing the vad_process() function and converts each element (audio chunk) of the queue to its 
            corresponding transcription. The threading was used to make sure both asr_process() and vad_process() run concurrently so that audio streaming and 
            its transcription both take place simultaneously.
        """
        
        wave2vec_asr = Wave2Vec2Inference(model_name)   # Wave2Vec2Inference from wav2vec2_inference.py is used to call its buffer_to_text() function (line-104).

        print("\nlistening to your voice\n")
        while True:                        
            audio_frames = in_queue.get()   # take the elements (audio chunks) of the element queue one by one.
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(  # re-scale the audio chunk
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()   # note time stamp
            text = wave2vec_asr.buffer_to_text(float64_buffer).lower()  # converts the audio chunk to its corresponding transcription.
            inference_time = time.perf_counter()-start
            sample_length = len(float64_buffer) / 16000  # length in sec
            if text != "":
                output_queue.put([text,sample_length,inference_time])  # put the text transcription in the output queue.                          

    def get_input_device_id(device_name, microphones):
        """ gets the input default device id and returns the first one."""
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    def list_microphones(pyaudio_instance):  # lists all the microphones in the device.
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()           

