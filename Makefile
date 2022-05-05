.PHONY: default run

NAME=tcia/masker
TAG=latest

container.last_built: Dockerfile src/masker/test.py src/masker/vtk_volume_rendering_DICOM.py src/masker/nbia_get_files_nlst.py requirements.txt src/masker/nbia_get_files_public.py
	docker build . -t ${NAME}:${TAG}
	date > container.last_built

run:
	docker run -it --rm -v ${PWD}../../data:/data ${NAME}:${TAG}

runtest: container.last_built
	docker run -it --rm -v /nas:/nas -v ${PWD}../../data:/data --entrypoint /run-test.sh ${NAME}:${TAG} 

shell:
	docker run -it --rm \
		-v /nas:/nas \
		-v ${PWD}../../data:/data \
		-v ${PWD}:/working \
		--entrypoint /working/shell.sh \
		${NAME}:${TAG} 

masker.sif:
	singularity build $@ docker-daemon://${NAME}:${TAG}
