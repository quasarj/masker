#!/bin/bash
# make
docker run \
    -it \
    --rm \
    -e POSDA_REDIS_HOST=144.30.104.84 \
    -e POSDA_REDIS_PORT=8382 \
    -e POSDA_API_URL=http://144.30.104.84:8383/papi \
	-e GRAYLOG_HOST=144.30.104.84 \
	-e GRAYLOG_PORT=8384 \
    -e FACE_EATER_DEBUG=1 \
    masker:latest
    # masker:latest --exit-on-empty
    # --entrypoint /bin/sh \
