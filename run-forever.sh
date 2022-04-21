#!/bin/bash
while true; do
docker run \
    -it \
    --rm \
    -e POSDA_REDIS_HOST=tcia-dev-2.ad.uams.edu \
    -e POSDA_API_URL=http://tcia-dev-2.ad.uams.edu/papi \
    -e FACE_EATER_DEBUG=True \
	-e GRAYLOG_HOST=tcia-graylog-1.ad.uams.edu \
	-e GRAYLOG_PORT=12201 \
    masker:latest
    # masker:latest --exit-on-empty
    # --entrypoint /bin/sh \
done
