#!/usr/bin/env bash

mkdir ./pretrained

wget https://www.dropbox.com/s/c5419pdvr0uq96m/generator_anime_faces.pt?dl=0 -O pretrained/G_anime.pt
wget https://www.dropbox.com/s/6917kb3rq2daije/generator_cifar10.pt?dl=0 -O pretrained/G_cifar.pt
wget https://www.dropbox.com/s/1lddrwgh6u0auy6/generator_lsun_bedroom.pt?dl=0 -O pretrained/G_lsun.pt

wget https://www.dropbox.com/s/qdm56175vkfbwnb/params_anime_faces.json?dl=0 -O pretrained/params_anime.json
wget https://www.dropbox.com/s/nvvgzoi969wp1l7/params_cifar10.json?dl=0 -O pretrained/params_cifar.json
wget https://www.dropbox.com/s/2yk61f05dp6ok44/params_lsun_bedroom.json?dl=0 -O pretrained/params_lsun.json
