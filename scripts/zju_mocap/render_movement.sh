#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

python run.py \
    --type movement \
    --cfg ./configs/human_nerf/zju_mocap/${SUBJECT}/adventure.yaml \
    load_net latest
