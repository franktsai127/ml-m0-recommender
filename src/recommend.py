import argparse
import pickle

import numpy as np

from preprocess import (
    load_data,
    build_ratings,
    build_watch_history,
)

from cold_start_recommend import recommend_cold_start_user

SIMILARITY_PATH = "models/item_similarity.pkl"
MOVIE_IDS_PATH = "models/movie_ids.pkl"


def load_model():
    with open(SIMILARITY_PATH, "rb") as f:
        similarity_matrix = pickle.load(f)

    with open(MOVIE_IDS_PATH, "rb") as f:
        movie_ids = pickle.load(f)

    return similarity_matrix, movie_ids


def recommend_existing_user(
    user_id,
    ratings,
    watched_by_user,
    movies,
    similarity_matrix,
    movie_ids,
    top_k=10,
):
    user_ratings = ratings[ratings["user_id"] == user_id]

    if user_ratings.empty:
        raise ValueError(f"User {user_id} has no rating history.")

    movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    rated_movie_ids = user_ratings["movie_id"].tolist()

    rated_indices = [
        movie_index[movie_id] for movie_id in rated_movie_ids if movie_id in movie_index
    ]

    rating_values = user_ratings[user_ratings["movie_id"].isin(movie_index)][
        "rating"
    ].to_numpy()

    # Ratings are from 1 to 10.
    # Center around the midpoint so high ratings contribute positively
    # and low ratings contribute negatively.
    centered_ratings = rating_values - 5.5

    candidate_similarities = similarity_matrix[:, rated_indices]

    scores = candidate_similarities @ centered_ratings

    watched_movies = watched_by_user.get(user_id, set())

    candidates = []

    for idx, movie_id in enumerate(movie_ids):
        if movie_id in watched_movies:
            continue

        candidates.append((movie_id, float(scores[idx])))

    candidates.sort(key=lambda x: x[1], reverse=True)

    top_candidates = candidates[:top_k]

    movie_info = movies.set_index("movie_id")

    recommendations = []

    for movie_id, score in top_candidates:
        movie = movie_info.loc[movie_id]

        recommendations.append(
            {
                "movie_id": movie_id,
                "title": movie["title"],
                "genres": movie["genres"],
                "score": score,
            }
        )

    return recommendations


def has_rating_history(user_id, ratings):
    return not ratings[ratings["user_id"] == user_id].empty


def main():
    parser = argparse.ArgumentParser(
        description="Generate movie recommendations for an existing user."
    )

    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="User ID to generate recommendations for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to return.",
    )

    args = parser.parse_args()

    events, users, movies = load_data()

    if args.user_id not in set(users["user_id"]):
        raise ValueError(f"User {args.user_id} does not exist.")

    ratings = build_ratings(events)

    watched_by_user = build_watch_history(events)

    if has_rating_history(
        args.user_id,
        ratings,
    ):
        similarity_matrix, movie_ids = load_model()

        print(f"\nUsing collaborative filtering " f"for existing user {args.user_id}.")

        recommendations = recommend_existing_user(
            user_id=args.user_id,
            ratings=ratings,
            watched_by_user=watched_by_user,
            movies=movies,
            similarity_matrix=similarity_matrix,
            movie_ids=movie_ids,
            top_k=args.top_k,
        )

        print(f"\nTop {args.top_k} recommendations " f"for user {args.user_id}:\n")

        for rank, rec in enumerate(
            recommendations,
            start=1,
        ):
            print(
                f"{rank}. {rec['title']} "
                f"[{rec['genres']}] "
                f"(score={rec['score']:.4f})"
            )

    else:
        print(
            f"\nUsing LLM-assisted cold-start "
            f"recommendation for user {args.user_id}."
        )

        profile, recommendations = recommend_cold_start_user(
            user_id=args.user_id,
            users=users,
            movies=movies,
            top_k=args.top_k,
        )

        print("\nStructured preference profile:")
        print(profile.model_dump_json(indent=2))

        print(f"\nTop {args.top_k} recommendations " f"for user {args.user_id}:\n")

        for rank, (_, movie) in enumerate(
            recommendations.iterrows(),
            start=1,
        ):
            print(
                f"{rank}. {movie['title']} "
                f"[{movie['genres']}] "
                f"(score={movie['score']:.4f})"
            )


if __name__ == "__main__":
    main()
