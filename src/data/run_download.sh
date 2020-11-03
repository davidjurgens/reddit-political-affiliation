#!/bin/sh

out='/shared/0/projects/reddit-political-affiliation/data/processed'
python3 download_flair_data.py $1 ${out}
