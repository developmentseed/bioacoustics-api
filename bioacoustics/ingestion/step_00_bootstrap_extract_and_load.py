import json
import logging
import os
import re
import concurrent.futures
import tempfile
import sys
import csv
import datetime

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

import utils

# Set this environment variable before loading the tensorflow
# library to squash warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = os.environ.get("MILVUS_DB_HOST", "localhost")
PORT = os.environ.get("MILVUS_DB_PORT", 19530)

COLLECTION_NAME = "a2o_bioacoustics"

DEFAULT_LOAD_PERCENTAGE = 100
BATCH_SIZE = 10000
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
    "filename",
)


def setup_collection(overwrite_collection_flag: bool = False):
    # # Drop collection if exists - PROCEED WITH CAUTION!!
    if overwrite_collection_flag and utility.has_collection(COLLECTION_NAME):
        logger.info("Collection exists, dropping...")
        collection = Collection(COLLECTION_NAME)
        collection.drop()

    # define collection fields
    id_field = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        descrition="primary field",
        is_primary=True,
        auto_id=True,
    )

    embedding_field = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        description="Float32 vector with dim 512",
        dim=512,
        is_primary=False,
    )

    file_timestamp_field = FieldSchema(
        name="file_timestamp",
        dtype=DataType.INT64,
        description="UTC File timestamp (in seconds since 1970-01-01T00:00:00)",
    )

    file_seconds_since_midnight_field = FieldSchema(
        name="file_seconds_since_midnight",
        dtype=DataType.INT64,
        description="Number of seconds since 00h00 (within same timezone)",
    )

    clip_offset_in_file_field = FieldSchema(
        name="clip_offset_in_file",
        dtype=DataType.INT64,
        description="Offset (in seconds) from start of file where embedding window starts",
    )

    channel_index_field = FieldSchema(
        name="chanel_index",
        dtype=DataType.INT64,
        description="Channel number for embedding",
    )

    # max len found in 1% sample = 4
    site_id_field = FieldSchema(
        name="site_id", dtype=DataType.VARCHAR, description="Site ID", max_length=8
    )

    # max len found in 1% sample = 50
    site_name_field = FieldSchema(
        name="site_name",
        dtype=DataType.VARCHAR,
        description="Site name",
        max_length=100,
    )

    # (one of: Dry-A, Dry-B, Wet-A, Wet-B)
    subsite_name_field = FieldSchema(
        name="subsite_name",
        dtype=DataType.VARCHAR,
        description="Subsite name",
        max_length=5,
    )

    # sequence id can be converted to int
    file_seq_id_field = FieldSchema(
        name="file_seq_id",
        dtype=DataType.INT64,
        description="File sequence ID",
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
            filename_field,
        ],
        description="Collection for A20 bird call embeddings",
    )

    collection = Collection(
        name=COLLECTION_NAME,
        # instantiate colleciton with no data
        data=None,
        schema=schema,
        # Set TTL to 0 to disable data expiration
        properties={"collection.ttl.seconds": 0},
    )

    logger.info(f"Collections: {utility.list_collections()}")

    logger.info("Creating new DISKANN index")
    # create new index (no build params required for DISKANN)
    index_params = {"index_type": "DISKANN", "metric_type": "L2", "params": {}}
    # index_params = {
    #     "index_type": "IVF_SQ8",
    #     "params": {"nlist": 1024},
    #     "metric_type": "L2",
    # }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


feature_description = {
    "timestamp_s": tf.io.FixedLenFeature([], tf.float32),
    "filename": tf.io.FixedLenFeature([], tf.string),
    "nuisances": tf.io.FixedLenFeature([], tf.string),
    "nuisances_shape": tf.io.FixedLenFeature([3], tf.int64),
    "embedding": tf.io.FixedLenFeature([], tf.string),
    "embedding_shape": tf.io.FixedLenFeature([3], tf.int64),
}


# Define a function to parse the TFRecordDataset
def parse_tfrecord(example_proto):
    # Parse the features from the serialized example
    features = tf.io.parse_single_example(example_proto, feature_description)

    # extract embedding as 3D array of float32, from byte string
    embedding = tf.io.parse_tensor(features["embedding"], out_type=tf.half)
    nuisances = tf.io.parse_tensor(features["nuisances"], out_type=tf.half)

    return (
        features["timestamp_s"],
        features["filename"],
        nuisances,
        features["nuisances_shape"],
        embedding,
        features["embedding_shape"],
    )


def extract_data(blob):
    with open("site_name_to_id_mapping.json", "r") as f:
        site_name_to_id_mapping = json.loads(f.read())
    with open("a2o_blocklist_no_sites.csv", "r") as f:
        blocklist = {b["filename"]: int(b["timestamp_s"]) for b in csv.DictReader(f)}

    count_unfiltered = 0
    count_blocklist_filtered = 0
    count_speech_filtered = 0
    count_empty_filtered = 0

    data = []
    with tempfile.NamedTemporaryFile() as tmpfile:
        blob.download_to_filename(tmpfile.name)

        raw_dataset = tf.data.TFRecordDataset(tmpfile.name)

        for (
            timestamp_s,
            filename,
            nuisances,
            nuisances_shape,
            embedding,
            embedding_shape,
        ) in raw_dataset.map(parse_tfrecord).as_numpy_iterator():
            filename = filename.decode("utf-8")

            if blocklist.get(filename) == int(timestamp_s):
                count_blocklist_filtered += 1
                continue

            [
                # (site_id, file_datetime, timezone, site_name, subsite_name, file_seq_id)
                (file_datetime, timezone, site_name, subsite_name, file_seq_id)
            ] = re.findall(
                # I'm quite proud of myself for this regex, but if anyone can see
                # a way to simplify it, please let me know!
                # r"site_(?P<site_id>\d{4})\/(?P<datetime>\d{8}T\d{6})(?P<timezone>(?:\+\d{4})|Z)_(?P<site_name>(?:\w*|-)*)-(?P<subsite_name>(?:Wet|Dry)-(?:A|B))_(?P<file_seq_id>\d*).flac",
                r"(?P<datetime>\d{8}T\d{6})(?P<timezone>(?:\+\d{4})|Z)_(?P<site_name>(?:\w*|-)*)-(?P<subsite_name>(?:Wet|Dry)-(?:A|B))_(?P<file_seq_id>\d*).flac",
                filename,
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
            # `nuisances` is a 3D array with Dims [12, 5, 3]
            # The first and second dimensions are identical to those of
            # the embeddings, and the 3 represents 3 different categories
            # of "issues" that may be present in the embedding: empty (no
            # animal calls), speech (human voices present in recording) and
            # unknown. The embeddings should be filtered out as follows:
            # if speech > - 0.25
            # if empty > 1.5 (only for the raw audio, or 0th channel)
            # if empty > 0.0 (only for the separated audio, or channels 1-4)

            for temporal_index, embedding_channels in enumerate(embedding):
                for channel_index, _embedding in enumerate(embedding_channels):
                    count_unfiltered += 1
                    if nuisances[temporal_index][channel_index][1] > -0.25:
                        count_speech_filtered += 1
                        continue
                    # Avoid indexing 'empty' channels.
                    if (
                        channel_index == 0
                        and nuisances[temporal_index][channel_index][0] > 1.5
                    ):
                        count_empty_filtered += 1
                        continue
                    # check if "empty" is greater than 0.0 for any of the separated audio channels
                    if (
                        channel_index != 0
                        and nuisances[temporal_index][channel_index][0] > 0.0
                    ):
                        count_empty_filtered += 1
                        continue

                    sanitized_full_site_name = (
                        f"{site_name.lower()}-{subsite_name.lower()}".replace(" ", "-")
                    )
                    site_id = site_name_to_id_mapping.get(sanitized_full_site_name)
                    if not site_id:
                        raise Exception(
                            f"No site id found for site: {sanitized_full_site_name}"
                        )

                    data.append(
                        {
                            "embedding": _embedding,
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
                            "filename": filename,
                            "nuisances": nuisances[temporal_index][
                                channel_index
                            ].tolist(),
                        }
                    )
            tmpfile.close()

    logging.info(
        f"Done processing {blob.name}. {len(data)} embeddings were processed (out of {count_unfiltered} attempted). Number of embeddings filtered: speech-{count_speech_filtered}, empty-{count_empty_filtered}. Number of files found in blocklis: {count_blocklist_filtered}"
    )

    return (
        data,
        count_unfiltered,
        count_blocklist_filtered,
        count_speech_filtered,
        count_empty_filtered,
    )


# helper function to batch data:
def split_into_batches(data, n=10_000):
    for i in range(0, len(data), n):
        yield data[i : i + n]


def load_into_miluvs(blob):
    logger.info(f"Processing blob: {blob.name}")

    (
        data,
        count_unfiltered,
        count_blocklist_filtered,
        count_speech_filtered,
        count_empty_filtered,
    ) = extract_data(blob)

    collection = Collection(COLLECTION_NAME)

    for batch in split_into_batches(data, BATCH_SIZE):
        collection.insert(
            [[_data[fieldname] for _data in batch] for fieldname in FIELD_NAMES]
        )

    return (
        len(data),
        count_unfiltered,
        count_blocklist_filtered,
        count_speech_filtered,
        count_empty_filtered,
    )


def main(
    overwrite_collection_flag: bool = False,
    load_percentage: float = DEFAULT_LOAD_PERCENTAGE,
):
    logger.info("Running updated 512-dim embeddings script with DiskANN")
    connections.connect(host=HOST, port=PORT)
    logger.info(f"Connections: {connections.list_connections()}")

    collection = setup_collection(overwrite_collection_flag)

    blobs = [
        blob
        for blob in utils.bucket.list_blobs(prefix=utils.EMBEDDINGS_FOLDER)
        if re.match(r"a2o_512/em_\d{3}/embeddings-\d{5}-of-\d{5}", blob.name)
    ]
    blobs = blobs[0:10]
    logger.info(f"Found {len(blobs)} blobs")

    # load only load_percentage of the files
    cutoff = max(2, int(len(blobs) * load_percentage / 100))
    blobs = blobs[:cutoff]

    logger.info(f"Loading data from {len(blobs)} blobs")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(lambda blob: load_into_miluvs(blob), blobs))

    # Only flush at the end of ingestion to ensure a new segment gets created
    # for any remaining vectors
    collection.flush()

    count_inserted = sum([r[0] for r in results])
    count_unfiltered = sum([r[1] for r in results])
    count_blocklist_filtered = sum([r[2] for r in results])
    count_speech_filtered = sum([r[3] for r in results])
    count_empty_filtered = sum([r[4] for r in results])

    logger.info(
        f"Collection {COLLECTION_NAME} currently loaded with {collection.num_entities} entities (out of {count_inserted} expected entities)"
    )
    logger.info(
        f"Total number of embeddings processed: {count_unfiltered}. Total number filtered: blocklist-{count_blocklist_filtered}, speech-{count_speech_filtered}, empty-{count_empty_filtered}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--overwrite-collection":
        overwrite_collection_flag = True
    else:
        overwrite_collection_flag = False
    main(overwrite_collection_flag, load_percentage=DEFAULT_LOAD_PERCENTAGE)
