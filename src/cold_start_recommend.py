import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import load_data
from cold_start import (
    get_cold_start_user,
    build_preference_profile,
)


def build_movie_text(movies):
    movie_text = (
        movies["title"].fillna("")
        + " "
        + movies["genres"].fillna("").str.replace("|", " ", regex=False)
        + " "
        + movies["overview"].fillna("")
    )

    return movie_text


def build_positive_query(profile):
    parts = []

    parts.extend(profile.liked_genres)
    parts.extend(profile.liked_keywords)
    parts.extend(profile.liked_movies)

    return " ".join(parts)


def build_negative_query(profile):
    parts = []

    parts.extend(profile.disliked_genres)
    parts.extend(profile.disliked_keywords)

    return " ".join(parts)


def recommend_cold_start_user(
    user_id,
    users,
    movies,
    top_k=10,
    negative_weight=0.75,
):
    likes, dislikes = get_cold_start_user(
        user_id,
        users,
    )

    profile = build_preference_profile(
        likes,
        dislikes,
    )

    movie_text = build_movie_text(movies)

    positive_query = build_positive_query(profile)
    negative_query = build_negative_query(profile)

    documents = movie_text.tolist()

    documents.append(positive_query)

    has_negative_query = bool(negative_query.strip())

    if has_negative_query:
        documents.append(negative_query)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    movie_vectors = tfidf_matrix[: len(movies)]

    positive_vector = tfidf_matrix[len(movies)]

    positive_scores = cosine_similarity(
        movie_vectors,
        positive_vector,
    ).flatten()

    if has_negative_query:
        negative_vector = tfidf_matrix[len(movies) + 1]

        negative_scores = cosine_similarity(
            movie_vectors,
            negative_vector,
        ).flatten()
    else:
        negative_scores = np.zeros(len(movies))

    disliked_genres = {genre.lower().strip() for genre in profile.disliked_genres}

    genre_penalties = np.zeros(len(movies))

    for idx, genres in enumerate(movies["genres"]):
        movie_genres = {genre.lower().strip() for genre in genres.split("|")}

        if movie_genres & disliked_genres:
            genre_penalties[idx] = 0.05

    final_scores = positive_scores - negative_weight * negative_scores - genre_penalties

    ranked_movies = movies.copy()

    ranked_movies["positive_score"] = positive_scores

    ranked_movies["negative_score"] = negative_scores

    ranked_movies["score"] = final_scores

    # Movies explicitly listed as liked were probably
    # already seen by the user, so exclude exact title matches.
    excluded_movie_titles = {
        title.lower().strip()
        for title in (profile.liked_movies + profile.disliked_movies)
    }

    ranked_movies = ranked_movies[
        ~ranked_movies["title"].str.lower().str.strip().isin(excluded_movie_titles)
    ]
    ranked_movies = ranked_movies.sort_values(
        "score",
        ascending=False,
    )

    recommendations = ranked_movies.head(top_k)

    return profile, recommendations


def main():
    parser = argparse.ArgumentParser(
        description=("Generate movie recommendations " "for a cold-start user.")
    )

    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="Cold-start user ID.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations.",
    )

    args = parser.parse_args()

    events, users, movies = load_data()

    profile, recommendations = recommend_cold_start_user(
        user_id=args.user_id,
        users=users,
        movies=movies,
        top_k=args.top_k,
    )

    print("\nStructured preference profile:")

    print(profile.model_dump_json(indent=2))

    print(
        f"\nTop {args.top_k} cold-start "
        f"recommendations for user "
        f"{args.user_id}:\n"
    )

    for rank, (_, movie) in enumerate(
        recommendations.iterrows(),
        start=1,
    ):
        print(
            f"{rank}. {movie['title']} "
            f"[{movie['genres']}] "
            f"(score={movie['score']:.4f}, "
            f"positive={movie['positive_score']:.4f}, "
            f"negative={movie['negative_score']:.4f})"
        )


if __name__ == "__main__":
    main()
