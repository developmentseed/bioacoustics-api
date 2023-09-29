import os
import logging
import re
import datetime
import json
import numpy as np
import tempfile

# Set this environment variable before loading the tensorflow
# library to squash warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

import utils


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

feature_description = {
    "timestamp_s": tf.io.FixedLenFeature([], tf.float32),
    "filename": tf.io.FixedLenFeature([], tf.string),
    "embedding": tf.io.FixedLenFeature([], tf.string),
    "embedding_shape": tf.io.FixedLenFeature([3], tf.int64),
}


# Define a function to parse the TFRecordDataset
def parse_tfrecord(example_proto):
    # Parse the features from the serialized example
    features = tf.io.parse_single_example(example_proto, feature_description)

    # extract embedding as 3D array of float32, from byte string
    embedding = tf.io.parse_tensor(features["embedding"], out_type=tf.float32)

    return (
        features["timestamp_s"],
        features["filename"],
        embedding,
        features["embedding_shape"],
    )


def process(blob):
    with open("site_name_to_id_mapping.json", "r") as f:
        site_name_to_id_mapping = json.loads(f.read())

    count = 0
    embeddings = []
    metadata = []
    with tempfile.NamedTemporaryFile(prefix="/data/") as tmpfile:
        blob.download_to_filename(tmpfile.name)

        raw_dataset = tf.data.TFRecordDataset(tmpfile.name)

        for timestamp_s, filename, embedding, embedding_shape in raw_dataset.map(
            parse_tfrecord
        ).as_numpy_iterator():
            [
                # (site_id, file_datetime, timezone, site_name, subsite_name, file_seq_id)
                (file_datetime, timezone, site_name, subsite_name, file_seq_id)
            ] = re.findall(
                # I'm quite proud of myself for this regex, but if anyone can see
                # a way to simplify it, please let me know!
                # r"site_(?P<site_id>\d{4})\/(?P<datetime>\d{8}T\d{6})(?P<timezone>(?:\+\d{4})|Z)_(?P<site_name>(?:\w*|-)*)-(?P<subsite_name>(?:Wet|Dry)-(?:A|B))_(?P<file_seq_id>\d*).flac",
                r"(?P<datetime>\d{8}T\d{6})(?P<timezone>(?:\+\d{4})|Z)_(?P<site_name>(?:\w*|-)*)-(?P<subsite_name>(?:Wet|Dry)-(?:A|B))_(?P<file_seq_id>\d*).flac",
                filename.decode("utf-8"),
            )

            # Some files have just "Z" as timezone, assume UTC in this case
            timezone = "+0000" if timezone == "Z" else timezone
            file_datetime = datetime.datetime.strptime(
                f"{file_datetime}{timezone}", "%Y%m%dT%H%M%S%z"
            )
            midnight = file_datetime.replace(hour=0, minute=0, second=0)
            file_offset_since_midnight = (file_datetime - midnight).seconds

            # `embedding` is a 3D array with Dims [12,5,1280]
            # The first dimension, [0:11] is the distinct 5 second windows
            # within a 60 second period.
            # The second dimension [0:4] is the 5 different audio channels
            # (4 separated + 1 combined audio channel)
            for temporal_index, embedding_channels in enumerate(embedding):
                for channel_index, _embedding in enumerate(embedding_channels):
                    sanitized_full_site_name = (
                        f"{site_name.lower()}-{subsite_name.lower()}".replace(" ", "-")
                    )
                    site_id = site_name_to_id_mapping.get(sanitized_full_site_name)
                    if not site_id:
                        raise Exception(
                            f"No site id found for site: {sanitized_full_site_name}"
                        )

                    count += 1

                    embeddings.append(_embedding)
                    metadata.append(
                        {
                            "file_timestamp": int(file_datetime.timestamp()),
                            "file_seconds_since_midnight": file_offset_since_midnight,
                            "recording_offset_in_file": int(
                                timestamp_s + (5 * temporal_index)
                            ),
                            "channel_index": channel_index,
                            "site_id": site_id,
                            "site_name": site_name,
                            "subsite_name": subsite_name,
                            "file_seq_id": int(file_seq_id),
                            "filename": filename.decode("utf-8"),
                        }
                    )
            tmpfile.close()

    # extract filename, removes extension
    stripped_filename = blob.name.split("/")[-1].split(".")[0]

    with tempfile.NamedTemporaryFile(prefix="/data/", mode="w") as tmpfile:
        tmpfile.write(json.dumps(metadata))
        metadata_blob = utils.bucket.blob(
            f"metadata_{utils.EMBEDDINGS_FOLDER}/{stripped_filename}.json"
        )
        metadata_blob.upload_from_filename(tmpfile.name)
        tmpfile.close()

    with tempfile.NamedTemporaryFile(prefix="/data/") as tmpfile:
        np.save(tmpfile, embeddings)
        numpy_blob = utils.bucket.blob(
            f"vector_{utils.EMBEDDINGS_FOLDER}/{stripped_filename}.npy"
        )
        numpy_blob.upload_from_filename(tmpfile.name)
        tmpfile.close()

    return len(embeddings)


def main():
    sample_data_blobs = [
        b
        for b in utils.storage_client.list_blobs(
            utils.BUCKET_NAME, prefix=utils.EMBEDDINGS_FOLDER
        )
    ]

    logger.info(f"{len(sample_data_blobs)} files to process found")

    total_count = 0
    for blob in sample_data_blobs:
        logger.info(f"Processing blob: {blob.name}")
        try:
            count = process(blob)
            total_count += count
        except Exception as e:
            logger.exception(f"Unable to process blob: {blob}")

    logger.info(f"Total number of data records: {total_count}")


if __name__ == "__main__":
    main()
