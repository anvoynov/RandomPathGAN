#!/usr/bin/env bash

mkdir ./pretrained

wget https://www.dropbox.com/s/c5419pdvr0uq96m/generator_anime_faces.pt?dl=0 -O pretrained/G_anime.pt
wget https://www.dropbox.com/s/6917kb3rq2daije/generator_cifar10.pt?dl=0 -O pretrained/G_cifar.pt
wget https://www.dropbox.com/s/1lddrwgh6u0auy6/generator_lsun_bedroom.pt?dl=0 -O pretrained/G_lsun.pt

wget https://www.dropbox.com/s/qdm56175vkfbwnb/params_anime_faces.json?dl=0 -O pretrained/params_anime.json
wget https://www.dropbox.com/s/nvvgzoi969wp1l7/params_cifar10.json?dl=0 -O pretrained/params_cifar.json
wget https://www.dropbox.com/s/2yk61f05dp6ok44/params_lsun_bedroom.json?dl=0 -O pretrained/params_lsun.json

# alternative LSUN train runs
wget https://www.dropbox.com/s/7malawxwoi7oy67/generator_lsun_2.pth?dl=0 -O pretrained/G_lsun_2.pt
wget https://www.dropbox.com/s/p67zxmp6p8whfb8/args_lsun_2.json?dl=0 -O pretrained/params_lsun_2.json

wget https://www.dropbox.com/s/d5k5ev3ve3djid6/generator_lsun_high_diversity.pth?dl=0 -O pretrained/G_lsun_high_diversity.pt
wget https://www.dropbox.com/s/qpsiptgkjf6x65o/args_lsun_high_diversity.json?dl=0 -O pretrained/params_lsun_high_diversity.json
