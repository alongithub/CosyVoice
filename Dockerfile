FROM python:3.8-slim

ENV TZ="Asia/Shanghai"
ENV TimeZone="Asia/Shanghai"
ENV LANG=C.UTF-8

COPY docker/sources.list /etc/apt/sources.list
COPY docker/resolv.conf /etc/resolv.conf
# RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free" > /etc/apt/sources.list \
#     && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free" >> /etc/apt/sources.list \
#     && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free" >> /etc/apt/sources.list \
#     && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list


RUN pip --version
RUN python --version

RUN ln -snf /usr/share/zoneinfo/ /etc/localtime && echo  > /etc/timezone && \
    rm -r /etc/apt/sources.list.d && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    libfst-dev \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
copy . .
RUN python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install pynini==2.1.5 -i https://mirrors.aliyun.com/pypi/simple/
RUN  pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN ln -s /usr/bin/python3 /usr/bin/python


WORKDIR /app/pretrained_models/CosyVoice-ttsfrd/
RUN  pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl
RUN export PYTHONPATH=third_party/Matcha-TTS

WORKDIR /app

cmd ["python", "cosy_api.py"]

