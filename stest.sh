#!/bin/bash
POSDA_REDIS_HOST=tcia-dev-2.ad.uams.edu \
POSDA_API_URL=http://tcia-dev-2.ad.uams.edu/papi \
FACE_EATER_DEBUG=True \
singularity run masker
