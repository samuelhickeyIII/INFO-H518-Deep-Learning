"""
Python Basics and Numpy - Assignment 1

Author: Samuel Hickey
Date:   9/4/2022

Problem Description: 
    Create a word-document matrix (A) from a text dataset. If the vocabulary
    (set of unique words) size is M and the number of documents is N, then
    the size of this matrix will be M X N. Use numpy data structures to create
    and manipulate this matrix.

Dataset:
    Twenty short-text files

Output:
    1.  Matrix A, Vocabulary size (M), Number of documents (N)
    2.  Use heatmap like visualization to display the word-document matrix
    3.  TF-IDF scores for each term in the vocabulary
    4.  The 3 documents found to be most similar to "10.txt" via cosine similarity
    5.  Using matrix manipulation and numpy create a new matrix B of size N X N,
            where Bij will represent the number of common words between documents
            i and j. Note: the diagonal of B should always have the highest value
"""
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from log.loggingSetup import setup_logging

def cleaner(text: str) -> list:
    '''
    Cleans a string

    text: str - a string to clean
    '''
    p = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789�'''
    return text.translate(str.maketrans('', '', p)) \
            .lower() \
            .split()


def cosine_similarity(A: np.array, B: np.array) -> float:
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def common_terms(A: np.array, B: np.array) -> int:
    count = 0
    for term in range(A.shape[0]):
        if A[term] != 0 and B[term] != 0:
            count += 1
    return int(count)

    
def frequency(terms: list[str], texts: list[list[str]]) -> np.matrix:
    return np.matrix(
        [np.array([text.count(term) for text in texts]) for term in terms]
    )


def load_documents(path: str) -> dict:
    '''
    Loads all files from a directory

    path: str - relative path to a directory
    '''
    docs: dict = dict()
    for document in os.listdir(path):
        with open(os.path.join(path, document), 'r') as f:
            docs[document] = cleaner(f.read())
    
    return docs


def tf_idf(mtx: np.matrix) -> np.matrix:
    tf, idf = np.zeros(mtx.shape), np.full((mtx.shape[0]), float(mtx.shape[1]))
    for term in range(mtx.shape[0]):
        for doc in range(mtx.shape[1]):
            tf[term, doc] = mtx[term, doc] / np.count_nonzero(mtx[:, doc])
        idf[term] = np.log10(idf[term] / np.count_nonzero(mtx[term, :]))

    return np.array([idf[term]*tf[term, :] for term in range(tf.shape[0])])

def main(logger):
    
    logger.info(f"Retrieving documents")
    files_dir: str = os.path.join(os.getcwd(), 'Assignments', 'A1', 'input')
    docs = load_documents(files_dir)

    logger.info("Combining lexicons")
    vocab = []
    for lexicon in docs.values(): vocab += lexicon
    vocab = sorted(list(set(vocab,)))[:-3]
    
    logger.info("Creating matrix 'A'")
    A = frequency(vocab, list(docs.values()))
    for row in A:
        logger.info(row)

    logger.info("Heatmap Visualization")
    fig = plt.figure(1, (10, 15))
    sns.heatmap(A, cmap=sns.cm.rocket_r)
    plt.show()

    logger.info("Calculating TF-IDF")
    scores = tf_idf(A)
    for term in scores:
        logger.info(term)

    logger.info("Calculating Cosine Similarity")
    cosine_scores = []
    for i in range(scores.shape[1]):
        if i == 9:
            continue
        cosine_scores.append((i, cosine_similarity(scores[9], scores[i])))
    print(sorted(cosine_scores, key=lambda x: x[1])[-3:])

    logger.info("Creating Matrix of Common Count")
    B = np.zeros((20,20), dtype= np.int16)
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            B[i, j] = common_terms(A[:, i], A[:, j])
    for row in B:
        print(row)


if __name__ == "__main__":
    try:
        setup_logging()
        logger = logging.getLogger("root")

        main(logger)

        exit(0)
    except Exception as e:
        logger.exception("A system exception has occurred: ")