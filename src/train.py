import os
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import (
    load_data,
    build_ratings,
    build_user_item_matrix,
)

MODEL_DIR = "models"
SIMILARITY_PATH = os.path.join(MODEL_DIR, "item_similarity.pkl")
MOVIE_IDS_PATH = os.path.join(MODEL_DIR, "movie_ids.pkl")


def build_item_similarity(user_item_matrix):
    # user_item_matrix:
    # rows    = users
    # columns = movies
    #
    # cosine_similarity expects each row to be one object.
    # We want to compare movies, so transpose:
    #
    # movie × user
    item_user_matrix = user_item_matrix.T.fillna(0.0)

    similarity_matrix = cosine_similarity(item_user_matrix)

    return similarity_matrix


def save_model(similarity_matrix, movie_ids):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(SIMILARITY_PATH, "wb") as f:
        pickle.dump(similarity_matrix, f)

    with open(MOVIE_IDS_PATH, "wb") as f:
        pickle.dump(movie_ids, f)


def main():
    events, users, movies = load_data()

    ratings = build_ratings(events)
    user_item_matrix = build_user_item_matrix(ratings)

    print("User-item matrix shape:")
    print(user_item_matrix.shape)

    similarity_matrix = build_item_similarity(user_item_matrix)

    movie_ids = user_item_matrix.columns.tolist()

    print("\nItem-item similarity matrix shape:")
    print(similarity_matrix.shape)

    print("\nSimilarity matrix value range:")
    print(
        float(np.min(similarity_matrix)),
        "to",
        float(np.max(similarity_matrix)),
    )

    print("\nFirst 5 movie IDs:")
    print(movie_ids[:5])

    print("\nExample similarities for first movie:")
    first_movie_id = movie_ids[0]
    first_movie_scores = similarity_matrix[0]

    top_indices = np.argsort(first_movie_scores)[::-1][:6]

    for idx in top_indices:
        print(movie_ids[idx], round(float(first_movie_scores[idx]), 4))

    save_model(similarity_matrix, movie_ids)

    print("\nSaved model files:")
    print(SIMILARITY_PATH)
    print(MOVIE_IDS_PATH)


if __name__ == "__main__":
    main()
