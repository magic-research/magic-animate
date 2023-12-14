FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/models

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    wget

RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /anaconda && \
    rm Anaconda3-2023.09-0-Linux-x86_64.sh

COPY environment.yaml /home/models/environment.yaml

ENV PATH="/home/models/.local/bin:/anaconda/bin:${PATH}"
ENV PATH="/anaconda/bin:${PATH}"
RUN echo "source /anaconda/etc/profile.d/conda.sh" >> /etc/profile
RUN /anaconda/bin/conda env create -f environment.yaml
ENV PATH="/anaconda/envs/manimate/bin:${PATH}"

RUN useradd -m -s /bin/bash models && echo "models:root" | chpasswd && adduser models sudo
RUN echo 'models ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY . .

RUN chown -R models:models /home/models
RUN chown -R 42420:42420 /home/models

EXPOSE 7860

CMD ["sh", "-c", "./entrypoint.sh"]

