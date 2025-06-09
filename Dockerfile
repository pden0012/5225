FROM public.ecr.aws/lambda/python:3.11-arm64


RUN yum -y install \
        mesa-libGL \
        libXrender \
        libXext \
        libsndfile \
        gcc \
        gcc-c++ \
        make \
        wget \
        tar \
        xz \
    && yum clean all


RUN cd /tmp && \
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz && \
    tar -xf ffmpeg-release-arm64-static.tar.xz && \
    cp ffmpeg-*-arm64-static/{ffmpeg,ffprobe} /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg* && \
    ffmpeg -version


RUN pip install --upgrade pip wheel setuptools


RUN pip install \
      numpy==1.23.5 \
      scipy==1.11.4 \
    --no-cache-dir


RUN pip install \
      soundfile==0.13.1 \
      soxr==0.5.0.post1 \
      audioread==3.0.1 \
      resampy==0.4.3 \
      pooch==1.8.2 \
      joblib==1.5.1 \
      decorator==5.2.1 \
    --no-cache-dir \
    --no-deps


RUN pip install librosa>=0.10.0 \
    --no-deps --no-cache-dir


RUN pip install \
      tensorflow==2.15.1 \
      tensorflow-estimator==2.15.0 \
      tensorflow-io-gcs-filesystem==0.37.1 \
    --no-cache-dir


COPY lambda_function.py birds_detection.py model.pt requirements.txt ./  
COPY BirdNET-Analyzer ./BirdNET-Analyzer


RUN pip install --no-deps -r requirements.txt --no-cache-dir

ENV NUMBA_DISABLE_JIT=1 \
    NUMBA_DISABLE_CACHING=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_THREADING_LAYER=safe \
    NUMBA_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/tmp \
    MPLCONFIGDIR=/tmp/.matplotlib


CMD ["lambda_function.lambda_handler"]