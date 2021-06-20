.PHONY: default run

NAME=masker
TAG=latest

default: Dockerfile
	docker build . -t ${NAME}:${TAG}


run:
	docker run -it --rm -v ${PWD}../../data:/data ${NAME}:${TAG}
