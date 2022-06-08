#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=monocular
fi

python run.py \
    --type movement \
    --cfg ./configs/human_nerf/wild/${SUBJECT}/adventure.yaml \
    load_net latest
