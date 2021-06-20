#!/bin/bash

export DISPLAY=:1

/usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf :1 &

echo "running with args: $@"
exec /app/masker/masker.py "$@"
