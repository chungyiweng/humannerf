#!/usr/bin/env bash

set -e

# Google drive links for pretrained models
declare -A gdrive_links
gdrive_links[377]=1kOZAdI5Qz9mTIfoM5MutZ23fzm_6_ZCr
gdrive_links[386]=1ecc7co8xRZsuUBELjaMSSdE50EB3iKsw
gdrive_links[387]=1WQoGgKxG2_wSMe0oPPWvSqQiNby0_uVH
gdrive_links[392]=1keEyz-tcr8ICHyRQ-WA42S_NzBvAvXVs
gdrive_links[393]=1m-No-69WBBsmCwsU2EQJr5x6AeIp_vUY
gdrive_links[394]=10S0iCr3KP2L6DGpqMjZ72sLp4H15FLGF

# Download the model
SUBJECT=$1
if [ -v gdrive_links[${SUBJECT}] ]
then
    EXP_DIR=experiments/human_nerf/zju_mocap/p${SUBJECT}/adventure
    mkdir -p ${EXP_DIR}
    gdown ${gdrive_links[${SUBJECT}]} -O ${EXP_DIR}/latest.tar
    echo "The downloaded model is at ${EXP_DIR}/latest.tar"
else
    echo "Subject ${SUBJECT}'s pretrained model does not exist."
fi

