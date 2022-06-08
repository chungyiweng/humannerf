#!/usr/bin/env bash

set -e

# Google drive links for pretrained models
declare -A gdrive_links
gdrive_links[377]=1QIBo5gKsrr9rohE0fex22kV0OJLaICUh
gdrive_links[386]=1WDZ9kLph43bLRv0eQB9iFkNYXE8RELZP
gdrive_links[387]=1LS_Ocw8LjHAlBalHNtJJSS0aHJqc3TKR
gdrive_links[392]=1dLUjpE_bN322S9U1isC8jHHged_Uf-rO
gdrive_links[393]=1dYcm4IWpV-UeXbjKlZYcZ1WTvg3KlJys
gdrive_links[394]=1fcJ0wAcm1Zk-Z_IBLTOPbPoHOgFc7rPZ

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

