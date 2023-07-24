from google.cloud import storage

BUCKET_NAME = "a20_dropbox"
EMBEDDINGS_FOLDER = f"{BUCKET_NAME}/one_percent_sep_embeddings"
PROJECT_ID = "dulcet-clock-385511"

storage_client = storage.Client(project=PROJECT_ID)

bucket=storage_client.bucket(BUCKET_NAME)