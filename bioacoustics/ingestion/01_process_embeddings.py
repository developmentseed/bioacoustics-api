import os
import re
import datetime
import json
import numpy as np

# Set this environment variable before loading the tensorflow 
# library to squash warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import utils

feature_description = {
    'timestamp_s': tf.io.FixedLenFeature([], tf.float32),
    'filename': tf.io.FixedLenFeature([], tf.string),
    'embedding': tf.io.FixedLenFeature([], tf.string),
    'embedding_shape': tf.io.FixedLenFeature([3], tf.int64)
}

# Define a function to parse the TFRecordDataset
def parse_tfrecord(example_proto):
    # Parse the features from the serialized example
    features = tf.io.parse_single_example(example_proto, feature_description)
    
    # extract embedding as 3D array of float32, from byte string 
    embedding = tf.io.parse_tensor(features["embedding"], out_type=tf.float32)
    
    return features['timestamp_s'], features["filename"], embedding, features["embedding_shape"]

if __name__ == "__main__": 

    sample_data_blobs = [b for b in utils.storage_client.list_blobs(utils.EMBEDDINGS_FOLDER)]

    print(f"{len(sample_data_blobs)} files to process found")

    count = 0
    for blob in sample_data_blobs: 
        local_filename = f"./{blob.name.split('/')[-1]}"
        blob.download_to_filename(local_filename)
        
        embeddings =[]
        metadata = []
        raw_dataset = tf.data.TFRecordDataset(local_filename)
        for timestamp_s, filename, embedding, embedding_shape in raw_dataset.map(parse_tfrecord).as_numpy_iterator():
            [(
            site_id, 
            file_datetime, 
            timezone, 
            site_name, 
            subsite_name, 
            file_seq_id
            )] = re.findall(
                # I'm quite proud of myself for this regex, but if anyone can see 
                # a way to simplify it, please let me know!
                r"site_(?P<site_id>\d{4})\/(?P<datetime>\d{8}T\d{6})(?P<timezone>(?:\+\d{4})|Z)_(?P<site_name>(?:\w*|-)*)-(?P<subsite_name>(?:Wet|Dry)-(?:A|B))_(?P<file_seq_id>\d*).flac",
                filename.decode("utf-8")
            )
            
            # Some files have just "Z" as timezone, assume UTC in this case
            timezone = "+0000" if timezone == "Z" else timezone
            file_datetime = datetime.datetime.strptime(f"{file_datetime}{timezone}", "%Y%m%dT%H%M%S%z")
            midnight = file_datetime.replace(hour=0, minute=0, second=0)
            file_offset_since_midnight = (file_datetime - midnight).seconds
            
            # `embedding` is a 3D array with Dims [12,5,1280]
            # The first dimension, [0:11] is the distinct 5 second windows
            # within a 60 second period.
            # The second dimension [0:4] is the 5 different audio channels
            # (4 separated + 1 combined audio channel)
            for temporal_index, embedding_channels in enumerate(embedding):
                for channel_index, _embedding in enumerate(embedding_channels): 
                    
                    count +=1

                    embeddings.append(_embedding)
                    metadata.append({
                        "file_timestamp": int(file_datetime.timestamp()),
                        "file_seconds_since_midnight": file_offset_since_midnight,
                        "recording_offset_in_file": int(timestamp_s + (5*temporal_index)), 
                        "channel_index": channel_index,
                        "site_id": site_id,
                        "site_name": site_name, 
                        "subsite_name": subsite_name, 
                        "file_seq_id": int(file_seq_id),
                        "filename": filename.decode("utf-8")
                    })
        
        # extract filename, removes extension
        stripped_filename = local_filename.split('.')[0]
        
        # prep filepaths for generated files
        metadata_filename = f"./{stripped_filename}.json"
        numpy_filename = f"./{stripped_filename}.npy"
        
        # write metadata to disk
        with open(metadata_filename, "w") as f: 
            f.write(json.dumps(metadata))
        
        # write embeddings to distk
        np.save(numpy_filename, embeddings)

        metadata_blob = utils.bucket.blob(f"{utils.EMBEDDINGS_FOLDER}/metadata/{metadata_filename}")
        metadata_blob.update_from_filename(metadata_filename)
        
        numpy_blob = utils.bucket.blob(f"{utils.EMBEDDINGS_FOLDER}/embeddings/{numpy_filename}")
        numpy_blob.update_from_filename(numpy_filename)


        # TODO: write `metadata_filename` to `gs://a20_dropbox/one_percent_sep_embeddings/metadata`
        # TODO: write `numpy_filename` to `gs://a20_dropbox/one_percent_sep_embeddings/embeddings`

        # NOTE: these two lines remove the files from local memory after they have been written 
        # to google cloud storage. This is important because there is about 350Gb worth of emebddings
        # in the gs bucket.
        os.remove(numpy_filename)
        os.remove(metadata_filename)
                
    print(f"Total number of data records: {count}")