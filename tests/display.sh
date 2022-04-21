#!/bin/bash

echo "Getting ids..."

ids=$(
http http://tcia-dev-2.ad.uams.edu/papi/v1/deface/$1 | \
	jq '.three_d_rendered_face,.three_d_rendered_face_box,.three_d_rendered_defaced'
)

for t in ${ids[@]}; do
	echo "displaying $t... close the window to continue"
	http http://tcia-dev-2.ad.uams.edu/papi/v1/files/$t/data | display
done
