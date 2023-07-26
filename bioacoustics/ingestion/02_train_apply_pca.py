import os
import numpy as np
import faiss
import utils
import tempfile

PCA_TRAINING_SAMPLE_SIZE = 0.25

if __name__ == "__main__":
    
    embeddings_blobs = [b for b in utils.storage_client.list_blobs(utils.EMBEDDINGS_FOLDER)]

    training_set = []

    for embeddings_blob in embeddings_blobs: 

        with tempfile.NamedTemporaryFile(prefix="/data") as tmpfile: 
            embeddings_blob.download_to_filename(tmpfile.name)
            embeddings = np.load(tmpfile.name)
        
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
    with tempfile.NamedTemporaryFile(prefix="/data") as tmpfile: 
    
        faiss.write_VectorTransform(pca_matrix, tmpfile.name)
        blob = utils.bucket.blob(f"{utils.EMBEDDINGS_FOLDER}_admin/1280_to_256_dimensionality_reduction.pca")
        blob.update_from_filename(tmpfile.name)


        
    for embeddings_blob in embeddings_blobs: 

        with tempfile.NamedTemporaryFile(prefix="/data") as tmpfile: 
            embeddings_blob.download_to_filename(tmpfile.name)
            embeddings = np.load(tmpfile.name)
        
        # apply the dimensionality reduction using the PCA matrix
        reduced_embeddings = pca_matrix.apply(embeddings)
        
        reduced_vector_blob_name = embeddings_blob.split("/")[-1]

        with tempfile.NamedTemporaryFile(prefix="/data") as tmpfile: 
            np.save(tmpfile.name, reduced_embeddings)
            blob = utils.bucket.blob(f"{utils.EMBEDDINGS_FOLDER}_reduced_vector/{reduced_vector_blob_name}")
            blob.update_from_filename(tmpfile.name)
        
        

       
       
