#!/usr/bin/env/bash

DATADIR='datasets/smoothing'
mkdir -p "$DATADIR"

#python3 ./scripts/smooth/get_data.py --out "$DATADIR" --url "https://drive.google.com/drive/folders/11hBQ_tZgFQKNdYy1d7WAKUoyWSr6FQfJ?usp=sharing"
cd "$DATADIR"
gdown https://drive.google.com/uc?id=1xfeamEaM_3MzGKK0lSbw0SF61be55utM
tar -xzvf basemesh.tar.gz && rm basemesh .tar.gz
wget https://www.dropbox.com/s/s3n05sw0zg27fz3/human_seg.tar.gz
tar -xzvf human_seg.tar.gz && rm human_seg.tar.gz
wget https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz
tar -xzvf shrec_16.tar.gz && rm shrec_16.tar.gz
find . -type f ! -name "*.obj" ! -name "*.stl" -delete
echo "Successfully downloaded data into :" $DATADIR
