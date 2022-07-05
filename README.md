# WCCAI STT Project

## Objective ##
The aim is to do live audio specch recognition (live STT) using facebook's Wav2Vec2 model. Here, live audio transcription is done by taking an audio stream with device microphone.

## Model Description ##
The model that's been exploited to perform speech-to-text in this case is facebook's Wav2Vec2. Initially facebook released its base model, which later was made available on huggingface website as an open-source AI model. Over the years, many developers have attempted fine-tuning the basic model with new and extensive datasets & techniques, resulting in a variety of pre-trained models for each language worldwide. 

The Wav2Vec2 model was chosen because it is one of the current state-of-the-art models for Automatic Speech Recognition owing to self-supervised training , which is a relatively novel idea in this sector. Also, the model architecture consists of many transformer layers, which gives it a cutting-edge touch. Then, the model can also be fine-tuned on any relatable dataset for a specific purpose.

I explored several fine-tuned Wav2Vec2 models available on the [huggingface model hub](https://huggingface.co/models) (especially issued by the verified developers) for the English language and found "facebook/wav2vec2-large-960h-lv60-self" as the aptest and most-accurate one. This is a large model pretrained and fine-tuned on 960 hours of Libri-Light and Librispeech datsets on 16kHz sampled speech audio. All the more, we use [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate) (WER) as a performance metric to determine this model's accuracy and efficiency. WER is a common metric of the performance of a speech recognition or machine translation system. The smaller the value of WER, the better the model performance. The test WER values for "facebook/wav2vec2-large-960h-lv60-self" are: 1.90 on LibriSpeech(clean) dataset and 3.90 on LibriSpeech(other) dataset, which suggests that it has outperformed many existing pre-trained models in this domain. Refer to [this](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) for more.

## Important details ##
The code and steps have been explained on the basis of its Python execution on a regular PC.\
I've already tested the Wav2Vec2 model on audio files, performing batch transcription i.e., transcription of the entire audio file in one shot with only a final result in a single line. Now is the time to undertake streaming transcription with audio through a microphone.

* Note that the model implemented is CPU extensive! I tried running it on both CPU and GPU processors, and the only difference noted is that with GPU, the program runs faster, which is also rather typical.

* All the Python Packages needed to be installed are given in the '[requirements.txt](https://github.com/harshrrajj/WCCAI-STT-Experiment/blob/main/requirements.txt)' file. Let's describe them one by one:
   * ***torch:*** The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors. This porgram uses PyTorch as an optimized Deep Learning tensor library based on Python and Torch.
   * ***torchaudio:*** Torchaudio is a library for audio and signal processing with PyTorch, which is generally used to provide powerful audio I/O functions.
   * ***transformers:*** Transformers package in Python contains a variety of pre-trained AI models, including the one we are using for this project. In our code, this package is simply used to download the "facebook/wav2vec2-large-960h-lv60-self" model.
   * ***pyaudio:*** PyAudio in Python is needed to play and record audio streams on a variety of platforms. In this project, it's just been used to collect an audio stream through the device microphone.
   * ***webrtcvad:*** WebRTC Voice Activity Detector (VAD) is a Python package that was made available by Google as an open-source model. It is used to classify a piece of audio data as being voiced or unvoiced. In speech recognition, the webrtcvad package is utilized to detect when the speaker is silent or takes a pause, which suggests to shift the transcription process to a new line.

* Possible issues with installing some of the packages and their resolution:
   * ***webrtcvad:***
     * **Issue:** Installation errors
     * **Resolution:** Do not use the pip command to install this package, rather use the following command on the anaconda command prompt: ```conda install -c conda-forge webrtcvad```
   * ***pyaudio:***
      * **Issue:** Installation errors
      * **Resolution:** Use the WHL file which can be downloaded from: [https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio). There are several versions that can be downloaded and used. For example, 'PyAudio‑0.2.11‑cp38‑cp38‑win_amd64.whl' is what I used for a 64-bit windows environment with Python version = 3.8. Then for installing the PyAudio package: Use the following command on the prompt: ```pip install "path"\PyAudio‑0.2.11‑cp38‑cp38‑win_amd64.whl```, where "path" is the location on your PC where you downloaded the WHL file.
   
   **Note:-** I've already handled the above-mentioned exceptions in the **DockerFile**, and it doesn't show any package-installation or other errors on execution. Therefore, if you are directly using the DockerFile to test the model, you do not need to be concerned about the above issues and resolutions.    

## Code walk-through ##
[live_asr.py](https://github.com/harshrrajj/WCCAI-STT-Experiment/blob/main/live-asr.py) is the heart of this project!\
All the code files (wav2vec2_inference.py, live_asr.py, and speechToTextLIVE.py) have been appropriately commented thoroughly. Every small detail can be found in the codes themselves, and they've been made easy to understand.

***To run the project and test speech-to-text, just execute the speechToTextLive.py file!***\
Or, run the [DockerFile](https://github.com/harshrrajj/WCCAI-STT-Experiment/blob/main/Dockerfile) directly. The docker file has been run on a Linux virtual environment and all the errors have been resolved.
