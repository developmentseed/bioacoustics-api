#! /bin/bash

# Exit immediately if any command fails
set -e

python 01_process_embeddings.py
python 02_train_apply_pca.py
python 03_bootstrap_milvus.py $1