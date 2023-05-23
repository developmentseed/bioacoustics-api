import os
import re
import datetime
import tensorflow as tf


EMBEDDINGS_DIR = "point_one_percent_embeddings"

# Define the feature description for parsing the TFRecordDataset
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

def prep_data_from_file(dataset_filename): 
    data = []
    raw_dataset = tf.data.TFRecordDataset(dataset_filename)
    count = 0
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
        
        # Some files have just "Z" as timezone
        timezone = "+0000" if timezone == "Z" else timezone
        file_datetime = datetime.datetime.strptime(f"{file_datetime}{timezone}", f"%Y%m%dT%H%M%S%z").timestamp()
        
        # `embedding` is a 3D array with Dims [12,1,1280]
        # We loop over the first dimension to "flatten" 
        # the 12 emebddings per minute
        # and extract the single channel (2nd dimension). 
        # We add each of the 12 embeddings as their own record
        for i, _embedding in enumerate(embedding[:,0]):
            _data = {
                "embedding": _embedding, 
                "file_timestamp": int(file_datetime), 
                "offset": int(timestamp_s + (5*i)), 
                "site_id": int(site_id), 
                "site_name": site_name, 
                "subsite_name": subsite_name, 
                "file_seq_id": int(file_seq_id),
                "filename": filename.decode("utf-8")
            }
            data.append(_data)
        count += 1
    print(f"Processed data file: {dataset_filename}, found {count} records")
    return data


def extract_sample_data(file_ids=[]): 
    if not os.path.exists(EMBEDDINGS_DIR): 
        os.system('gsutil -m cp -r "gs://a20_dropbox/point_one_percent_embeddings" .')
    data = [
        prep_data_from_file(f"point_one_percent_embeddings/a2o_sample_embeddings-0000{i}-of-00007") 
        for i in (file_ids if file_ids else range(0,7))
    ]

    # flatten 2 list for of records 
    flattened = [record for _data in data for record in _data]
    return [{**r, "id":i} for i, r in enumerate(flattened)]

# helper function to batch data: 
def split_into_batches(data, n=10_000): 
    for i in range(0, len(data), n):
        yield data[i:i + n]
 