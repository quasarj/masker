#!/bin/bash

export DISPLAY=:1

/usr/bin/Xorg \
	-noreset \
	+extension GLX \
	+extension RANDR \
	+extension RENDER \
	-logfile ./xdummy.log \
	-config /etc/X11/xorg.conf \
	:1 \
	>/dev/null 2>/dev/null &

echo "running with args: $@"
exec python3 /app/masker/masker.py "$@"
