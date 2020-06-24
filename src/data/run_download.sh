#!/bin/sh

out='/home/kalkiek/projects/reddit-political-affiliation/data/processed/'
python download_flair_data.py $1 ${out}
