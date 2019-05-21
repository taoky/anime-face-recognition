from ubuntu:18.04

RUN sed -i "s/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && sed -i "s/http:\/\/security.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y libmagickwand-dev build-essential ruby-dev && gem install rmagick

COPY animeface-2009/ /animeface-2009/
WORKDIR /animeface-2009/
RUN ./build.sh

RUN bash
