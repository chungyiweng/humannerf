#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

FREE_VIEW_FRAME_IDX=$2
if [ -z "${FREE_VIEW_FRAME_IDX}" ]
then
    FREE_VIEW_FRAME_IDX=0
fi

python run.py \
    --type freeview \
    --cfg ./configs/human_nerf/zju_mocap/${SUBJECT}/adventure.yaml \
    load_net latest \
    freeview.frame_idx ${FREE_VIEW_FRAME_IDX}
