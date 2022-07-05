# base image
FROM python:3.9-slim-buster

# username argument read from CLI
ARG USERNAME

# execute these commands as root user
RUN apt-get update && \
    apt-get install --no-install-recommends -y gcc libasound-dev libsndfile1-dev libportaudio2 libportaudiocpp0 portaudio19-dev && \
    pip3 install --upgrade pip && \
    useradd -rm ${USERNAME} && \
    mkdir /home/${USERNAME}/speechToTextLIVE

# change working directory
WORKDIR /home/${USERNAME}/speechToTextLIVE

# copy python script to current directory
COPY live_asr.py .
COPY requirements.txt .
COPY speechToTextLIVE.py .
COPY wav2vec2_inference.py .

# execute these commands as root user
# grant executable permission to speechToTextLIVE.py
# change owner of home directory
RUN pip3 install -r requirements.txt && \
    chmod +x speechToTextLIVE.py && \
    chown -R ${USERNAME}: /home/${USERNAME}

# change active user
USER ${USERNAME}

# execute this command when container boots
CMD ./speechToTextLIVE.py
