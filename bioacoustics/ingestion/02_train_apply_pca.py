import os
import numpy as np
import faiss

PCA_TRAINING_SAMPLE_SIZE = 0.25

if __name__ == "__main__":
    embeddings_files = [
    # TODO: LIST contents of: `gs://a20_dropbox/one_percent_sep_embeddings/embeddings`
    ]

    training_set = []

    for embedding_file in embeddings_files: 

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

    # write to file
    faiss.write_VectorTransform(mat, "1280_to_256_dimensionality_reduction.pca")

    # TODO: write `./1280_to_256_dimensionality_reduction.pca` to a location in 
    # google cloud storage that the API can access

    # read trained PCA matrix from file
    pca_matrix = faiss.read_VectorTransform("1280_to_256_dimensionality_reduction.pca")

    for embedding_file in embeddings_files: 
        
        embeddings = np.load(embedding_file)
        
        # apply the dimensionality reduction using the PCA matrix
        reduced_embeddings = pca_matrix.apply(embeddings)
        
        # write the reduced embeddings to disk
        filename = embedding_file.split("/")[-1]
        
        np.save(f"./{filename}", reduced_embeddings)

        # TODO: write `filename` to `gs://a20_dropbox/one_percent_sep_embeddings/reduced`
