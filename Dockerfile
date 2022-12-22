FROM python:3.9

COPY requirements.txt ./requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge

COPY ./ /app/
WORKDIR /app/

CMD streamlit run src/cdc_app.py