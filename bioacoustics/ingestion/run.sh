#! /bin/bash    

python 01_process_embeddings.py
python 02_train_apply_pca.py
python 03_bootstrap_milvus.py