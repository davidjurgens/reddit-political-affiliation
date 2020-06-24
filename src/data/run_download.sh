#!/bin/sh

out='/home/kalkiek/projects/reddit-political-affiliation/data/processed/'
python3 download_flair_data.py $1 ${out}
