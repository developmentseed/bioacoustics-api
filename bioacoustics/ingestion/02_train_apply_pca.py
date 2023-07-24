import os
import numpy as np
import faiss
import utils

PCA_TRAINING_SAMPLE_SIZE = 0.25

if __name__ == "__main__":
    
    embeddings_blobs = [b for b in utils.storage_client.list_blobs(utils.EMBEDDINGS_FOLDER)]

    training_set = []

    for embeddings_blob in embeddings_blobs: 

        embedding_file = f"./{embeddings_blob.name.split('/')[-1]}"
        embeddings_blob.download_to_filename(embedding_file)

        embeddings = np.load(embedding_file)
        
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
    mat = faiss.PCAMatrix(1280, 256)

    # train
    mat.train(training_set)

    trained_pca_filename = "1280_to_256_dimensionality_reduction.pca"

    # write to file
    faiss.write_VectorTransform(mat, trained_pca_filename)

    blob = utils.bucket.blob(f"{utils.EMBEDDINGS_FOLDER}/{trained_pca_filename}")
    blob.update_from_filename(trained_pca_filename)


    # read trained PCA matrix from file
    pca_matrix = faiss.read_VectorTransform(trained_pca_filename)

    for embeddings_blob in embeddings_blobs: 

        embedding_file = f"./{embeddings_blob.name.split('/')[-1]}"
        embeddings_blob.download_to_filename(embedding_file)
        
        embeddings = np.load(embedding_file)
        
        # apply the dimensionality reduction using the PCA matrix
        reduced_embeddings = pca_matrix.apply(embeddings)
        
        # write the reduced embeddings to disk
        filename = embedding_file.split("/")[-1]
        
        np.save(f"./{filename}", reduced_embeddings)

        blob = utils.bucket.blob(f"{utils.EMBEDDINGS_FOLDER}/reduced/{filename}")
        blob.update_from_filename(filename)

