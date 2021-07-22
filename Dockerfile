FROM python:3

ENV DEBIAN_FRONTEND noninteractive
ENV DISPLAY :1

RUN apt-get update \
    && apt-get -y install xserver-xorg-video-dummy x11-apps python3 cmake


COPY requirements.txt /
COPY src /app
RUN pip install -r /requirements.txt

COPY xorg.conf /etc/X11/xorg.conf
COPY run.sh /run.sh

ENTRYPOINT ["/run.sh"]
