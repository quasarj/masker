#FROM debian:jessie
FROM python:3

ENV DEBIAN_FRONTEND noninteractive
ENV DISPLAY :1

RUN apt-get update \
    && apt-get -y install xserver-xorg-video-dummy x11-apps python3 cmake


COPY src /app
RUN pip install -r /app/masker/requirements.txt


VOLUME /tmp/.X11-unix

COPY xorg.conf /etc/X11/xorg.conf

COPY run.sh /run.sh


# ENTRYPOINT "/run.sh"
CMD "/run.sh"
#CMD ["/usr/bin/Xorg", "-noreset", "+extension", "GLX", "+extension", "RANDR", "+extension", "RENDER", "-logfile", "./xdummy.log", "-config", "/etc/X11/xorg.conf", ":1"]
