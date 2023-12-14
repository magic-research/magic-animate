FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/models

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    wget \
    sudo \
    openssh-server \
    fail2ban \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg-agent \
    libgl1 \
    libglib2.0-0 \
    lshw \
    libtcmalloc-minimal4 \
    apt-utils

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

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

EXPOSE 7860 22

CMD ["sh", "-c", "service ssh start && tail -f /dev/null"]

#ssh -L 7860:127.0.0.1:7860 -p 2222 models@localhost

