import logging
import numpy as np
import faiss
import utils
import tempfile

PCA_TRAINING_SAMPLE_SIZE = 0.05

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting PCA training...")
    embeddings_blobs = [
        b for b in utils.storage_client.list_blobs(utils.BUCKET_NAME, prefix=f"vector_{utils.EMBEDDINGS_FOLDER}")
    ]

    training_set = []

    for embeddings_blob in embeddings_blobs: 

        with tempfile.NamedTemporaryFile(prefix="/data/") as tmpfile: 
            embeddings_blob.download_to_filename(tmpfile.name)
            embeddings = np.load(tmpfile.name, allow_pickle=True)
        
        # select random subset by generated random indexes to select
        rand_indexes = np.random.randint(
            low=0, high=len(embeddings), size=int(PCA_TRAINING_SAMPLE_SIZE * len(embeddings))
        )
        subset = embeddings[rand_indexes]
        
        # use native python list for append to the training set 
        # (since you can't append to numpy arrays inplace)
        training_set.extend(list(subset))

    # convert training set to numpy array for compatibility
    # with the faiss PCA module
    training_set = np.array(training_set)

    # define PCA matrix
    pca_matrix = faiss.PCAMatrix(1280, 256)

    # train
    pca_matrix.train(training_set)

    # write to file
    with tempfile.NamedTemporaryFile(prefix="/data/") as tmpfile: 
    
        faiss.write_VectorTransform(pca_matrix, tmpfile.name)
        blob = utils.bucket.blob(f"admin_{utils.EMBEDDINGS_FOLDER}/1280_to_256_dimensionality_reduction.pca")
        blob.upload_from_filename(tmpfile.name)

    logger.info("Finished PCA training ...")


        
    for embeddings_blob in embeddings_blobs: 
        logger.info(f"Applying dimensionality reduction to {embeddings_blob.name}")
        with tempfile.NamedTemporaryFile(prefix="/data/") as tmpfile: 
            embeddings_blob.download_to_filename(tmpfile.name)
            embeddings = np.load(tmpfile.name, allow_pickle=True)
        
        # apply the dimensionality reduction using the PCA matrix
        reduced_embeddings = pca_matrix.apply(embeddings)
        
        reduced_vector_blob_name = embeddings_blob.name.split("/")[-1]

        with tempfile.NamedTemporaryFile(prefix="/data/") as tmpfile: 
            np.save(tmpfile, reduced_embeddings)
            blob = utils.bucket.blob(f"reduced_vector_{utils.EMBEDDINGS_FOLDER}/{reduced_vector_blob_name}")
            blob.upload_from_filename(tmpfile.name)
