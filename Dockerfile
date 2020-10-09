FROM debian:buster
FROM python:3.7-buster

RUN echo 'deb http://deb.debian.org/debian buster main contrib non-free' >> /etc/apt/sources.list && cat /etc/apt/sources.list
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           apt-utils \
           bzip2 \
           ca-certificates \
           curl \
           locales \
           unzip \
           git \
           cmake \
    && apt-get clean

ENV ANTSPATH="/opt/ants" \
    PATH="/opt/ants:$PATH" \
    CMAKE_INSTALL_PREFIX=$ANTSPATH

RUN echo "Cloning ANTs repo..." \
    && mkdir ~/code \
    && cd ~/code \
    && git clone https://github.com/ANTsX/ANTs.git

RUN cd ~/code/ANTs && git checkout tags/v2.3.1

RUN echo "Building ANTs..." \
    && mkdir -p ~/bin/antsBuild \
    && cd ~/bin/antsBuild \
    && cmake ~/code/ANTs
RUN cd ~/bin/antsBuild/ \
    && make -j4
RUN mv ~/bin/antsBuild/bin $ANTSPATH

COPY ./requirements.txt /pattools/requirements.txt
WORKDIR /pattools
RUN pip3 install -r requirements.txt --src /usr/local/src
COPY ./pattools /pattools/pattools
COPY ./*.py /pattools/
COPY ./*.cfg /pattools/
COPY ./README.md /pattools/

RUN python3 setup.py install