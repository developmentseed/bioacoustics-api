import os
from google.cloud import storage


BUCKET_NAME = os.environ.get("EMBEDDINGS_BUCKET", "a20_dropbox")
EMBEDDINGS_FOLDER = os.environ.get("EMBEDDINGS_FOLDER", "one_percent_sep_embeddings")
PROJECT_ID = os.environ.get("PROJECT_ID", "bioacoustics-216319")

storage_client = storage.Client(project=PROJECT_ID)

bucket=storage_client.bucket(BUCKET_NAME)