FROM python:3.9-bullseye

LABEL PRODUCT="TebakAja"
LABEL SERVICE="Cryptocurrency Service"
LABEL TEAM="System and Machine Learning Engineering Team"


RUN useradd -m -u 1000 user

WORKDIR /app


# Install Requirements
RUN apt-get update && \
    apt-get install -y gcc python3-dev gnupg curl wget

COPY --chown=user ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app


# Cythonizing Utilities
RUN pip install cython

RUN cd /app/restful/cutils && \
    python setup.py build_ext --inplace && \
    chmod 777 * && cd ../..


# Initialization Resources
RUN mkdir -p /app/resources && \
    chmod 755 /app/resources


# Datasets Resources
RUN mkdir -p /app/resources/ && \
    chmod 755 /app/resources/
    
RUN --mount=type=secret,id=DATASETS_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/DATASETS_ID) \
            -O /app/resources/datasets.zip && \
    unzip -o /app/resources/datasets.zip -d /app/resources/ && \
    rm /app/resources/datasets.zip

RUN ls /app/resources/datasets


# Algorithms Resources
RUN mkdir -p /app/resources/algorithms && \
    chmod 755 /app/resources/algorithms


# GRU Algorithm Resources
RUN mkdir -p /app/resources/algorithms/GRU && \
    chmod 755 /app/resources/algorithms/GRU

RUN --mount=type=secret,id=GRU_MODELS_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/GRU_MODELS_ID) \
            -O /app/resources/algorithms/GRU/models.zip && \
    unzip -o /app/resources/algorithms/GRU/models.zip \
        -d /app/resources/algorithms/GRU && \
    rm /app/resources/algorithms/GRU/models.zip

RUN --mount=type=secret,id=GRU_PICKLES_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/GRU_PICKLES_ID) \
            -O /app/resources/algorithms/GRU/pickles.zip && \
    unzip -o /app/resources/algorithms/GRU/pickles.zip \
        -d /app/resources/algorithms/GRU && \
    rm /app/resources/algorithms/GRU/pickles.zip

RUN --mount=type=secret,id=GRU_POSTTRAINED_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/GRU_POSTTRAINED_ID) \
            -O /app/resources/algorithms/GRU/posttrained.zip && \
    unzip -o /app/resources/algorithms/GRU/posttrained.zip \
        -d /app/resources/algorithms/GRU && \
    rm /app/resources/algorithms/GRU/posttrained.zip


# LSTM Algorithm Resources
RUN mkdir -p /app/resources/algorithms/LSTM && \
    chmod 755 /app/resources/algorithms/LSTM

RUN --mount=type=secret,id=LSTM_MODELS_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/LSTM_MODELS_ID) \
            -O /app/resources/algorithms/LSTM/models.zip && \
    unzip -o /app/resources/algorithms/LSTM/models.zip \
        -d /app/resources/algorithms/LSTM && \
    rm /app/resources/algorithms/LSTM/models.zip

RUN --mount=type=secret,id=LSTM_PICKLES_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/LSTM_PICKLES_ID) \
            -O /app/resources/algorithms/LSTM/pickles.zip && \
    unzip -o /app/resources/algorithms/LSTM/pickles.zip \
        -d /app/resources/algorithms/LSTM && \
    rm /app/resources/algorithms/LSTM/pickles.zip

RUN --mount=type=secret,id=LSTM_POSTTRAINED_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/LSTM_POSTTRAINED_ID) \
            -O /app/resources/algorithms/LSTM/posttrained.zip && \
    unzip -o /app/resources/algorithms/LSTM/posttrained.zip \
        -d /app/resources/algorithms/LSTM && \
    rm /app/resources/algorithms/LSTM/posttrained.zip


# LSTM_GRU Algorithm Resources
RUN mkdir -p /app/resources/algorithms/LSTM_GRU && \
    chmod 755 /app/resources/algorithms/LSTM_GRU

RUN --mount=type=secret,id=LSTM_GRU_MODELS_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/LSTM_GRU_MODELS_ID) \
            -O /app/resources/algorithms/LSTM_GRU/models.zip && \
    unzip -o /app/resources/algorithms/LSTM_GRU/models.zip \
        -d /app/resources/algorithms/LSTM_GRU && \
    rm /app/resources/algorithms/LSTM_GRU/models.zip

RUN --mount=type=secret,id=LSTM_GRU_PICKLES_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/LSTM_GRU_PICKLES_ID) \
            -O /app/resources/algorithms/LSTM_GRU/pickles.zip && \
    unzip -o /app/resources/algorithms/LSTM_GRU/pickles.zip \
        -d /app/resources/algorithms/LSTM_GRU && \
    rm /app/resources/algorithms/LSTM_GRU/pickles.zip

RUN --mount=type=secret,id=LSTM_GRU_POSTTRAINED_ID,mode=0444,required=true \
    wget -q --show-progress --progress=bar:force \
        https://drive.google.com/uc?id=$(cat /run/secrets/LSTM_GRU_POSTTRAINED_ID) \
            -O /app/resources/algorithms/LSTM_GRU/posttrained.zip && \
    unzip -o /app/resources/algorithms/LSTM_GRU/posttrained.zip \
        -d /app/resources/algorithms/LSTM_GRU && \
    rm /app/resources/algorithms/LSTM_GRU/posttrained.zip



CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--workers", "30", "--port", "7860"]

