import json
import numpy as np
import concurrent.futures
import utils
import tempfile
import sys
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

HOST = "" # TODO 
PORT = 0 # TODO

COLLECTION_NAME = "a2o_bioacoustics"

BATCH_SIZE=1000
FIELD_NAMES = (
    "embedding",
    "file_timestamp",
    "file_seconds_since_midnight",
    "recording_offset_in_file",
    "channel_index",
    "site_id",
    "site_name",
    "subsite_name", 
    "file_seq_id", 
    "filename"
)

def setup_collection(overwrite_collection_flag:bool = False):
    # # Drop collection if exists - PROCEED WITH CAUTION!!
    if overwrite_collection_flag and utility.has_collection(COLLECTION_NAME): 
        print("Collection exists, dropping...")
        collection = Collection(COLLECTION_NAME)
        collection.drop()

    # define collection fields
    id_field = FieldSchema(
        name="id", dtype=DataType.INT64, descrition="primary field", is_primary=True, auto_id=True
    )

    embedding_field = FieldSchema(
        name="embedding", 
        dtype=DataType.FLOAT_VECTOR, 
        description="Float32 vector with dim 256 (reduced using PCA from origina vector dim:1280)", 
        dim=256,
        is_primary=False
    )

    file_timestamp_field = FieldSchema(
        name="file_timestamp", 
        dtype=DataType.INT64, 
        description="UTC File timestamp (in seconds since 1970-01-01T00:00:00)"
    )

    file_seconds_since_midnight_field = FieldSchema(
        name="file_seconds_since_midnight", 
        dtype=DataType.INT64, 
        description="Number of seconds since 00h00 (within same timezone)"
    )

    clip_offset_in_file_field = FieldSchema(
        name="clip_offset_in_file", 
        dtype=DataType.INT64, 
        description="Offset (in seconds) from start of file where embedding window starts"
    )

    channel_index_field = FieldSchema(
        name="chanel_index", dtype=DataType.INT64, description="Channel number for embedding"
    )

    # max len found in 1% sample = 4
    site_id_field = FieldSchema(
        name="site_id", dtype=DataType.VARCHAR, description="Site ID", max_length=8
    )

    # max len found in 1% sample = 50
    site_name_field = FieldSchema(
        name="site_name", dtype=DataType.VARCHAR, description="Site name", max_length=100 
    )

    # (one of: Dry-A, Dry-B, Wet-A, Wet-B)
    subsite_name_field = FieldSchema(
        name="subsite_name", dtype=DataType.VARCHAR, description="Subsite name", max_length=5 
    )

    # sequence id can be converted to int
    file_seq_id_field = FieldSchema(
        name="file_seq_id", dtype=DataType.INT64, description="File sequence ID", 
    )
    filename_field = FieldSchema(
        name="filename", dtype=DataType.VARCHAR, max_length=500
    )

    schema = CollectionSchema(
        fields=[
            id_field,
            embedding_field, 
            file_timestamp_field,
            file_seconds_since_midnight_field,
            clip_offset_in_file_field, 
            channel_index_field,
            site_id_field, 
            site_name_field, 
            subsite_name_field, 
            file_seq_id_field, 
            filename_field
        ], 
        description="Collection for A20 bird call embeddings"
    )


    collection = Collection(
        name=COLLECTION_NAME, 
        # instantiate colleciton with no data
        data=None,
        schema=schema, 
        # Set TTL to 0 to disable data expiration
        properties={"collection.ttl.seconds": 0}
    )

    print(f"Collections: {utility.list_collections()}")

    print(f"Creating new index: IVF_SQ8 with params nlist:4096")
    # create new index: 
    index_params = {
        "index_type": "IVF_SQ8",
        "params":{"nlist": 4096},
        "metric_type": "L2"
    }

    collection.create_index("embedding", index_params)
    return collection


# helper function to batch data: 
def split_into_batches(data, n=10_000): 
    for i in range(0, len(data), n):
        yield data[i:i + n]

def load_data(metadata_blob): 
    
    with tempfile.NamedTemporaryFile(prefix="/data") as tmpfile: 
        metadata_blob.download_to_filename(tmpfile.name)
        with open(tmpfile.name, "r") as f: 
            _metadata = json.loads(f.read())    

    
    embeddings_blobname = metadata_blob.name.split("/")[-1].replace(".json", ".npy")
    embedddings_blob = utils.bucket.blob(f"{utils.EMBEDDINGS_FOLDER}_reduced_vector/{embeddings_blobname}")
    
    with tempfile.NamedTemporaryFile(prefix="/data") as tmpfile: 
        embedddings_blob.download_to_filename(tmpfile.name)
        _embeddings = np.load(tmpfile.name)
    
    assert len(_embeddings) == len(_metadata)
    
    data = [
        {"embedding": _embeddings[i], **_metadata[i]} 
        for i in range(len(_metadata)) 
    ]
        
    collection = Collection(COLLECTION_NAME)
    
    for batch in split_into_batches(data, BATCH_SIZE): 
        
        collection.insert(
            [[_data[fieldname] for _data in batch] for fieldname in FIELD_NAMES]
        )
        collection.flush()

    return len(_embeddings)

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1].lower() == "--overwrite-collection":
        overwrite_collection_flag = True
    else:
        overwrite_collection_flag = False

    connections.connect(host=HOST, port=PORT)
    print("Connections: ", connections.list_connections())

    collection = setup_collection(overwrite_collection_flag)

    metadata_blobs = [b for b in utils.storage_client.list_blobs(f"{utils.EMBEDDINGS_FOLDER}_metadata")]
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(
            executor.map(
                lambda x: load_data(x),
                metadata_blobs
            ), 
            total=len(metadata_blobs) 
        
        ) 

    print(f"Collection {COLLECTION_NAME} currently loaded with {collection.num_entities} entities")
        