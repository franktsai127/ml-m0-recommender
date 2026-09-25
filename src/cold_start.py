import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from preprocess import load_data

load_dotenv()


class PreferenceProfile(BaseModel):
    liked_genres: list[str]
    liked_keywords: list[str]
    liked_movies: list[str]
    disliked_genres: list[str]
    disliked_keywords: list[str]
    disliked_movies: list[str]


def get_cold_start_user(user_id, users):
    user = users[users["user_id"] == user_id]

    if user.empty:
        raise ValueError(f"User {user_id} does not exist.")

    user = user.iloc[0]

    likes = user["self_description_likes"]
    dislikes = user["self_description_dislikes"]

    if not isinstance(likes, str) or not isinstance(dislikes, str):
        raise ValueError(
            f"User {user_id} does not have complete self-description data."
        )

    return likes, dislikes


def build_preference_profile(likes, dislikes):
    client = OpenAI()

    model = os.getenv("OPENAI_MODEL", "gpt-5.6")

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You convert a movie user's free-form likes and dislikes "
                    "into a concise structured preference profile. "
                    "Use common movie genre names where possible. "
                    "Extract only preferences supported by the user's text. "
                    "Do not invent preferences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Movies the user likes:\n{likes}\n\n"
                    f"Movies the user dislikes:\n{dislikes}"
                ),
            },
        ],
        text_format=PreferenceProfile,
    )

    return response.output_parsed


def main():
    parser = argparse.ArgumentParser(
        description="Generate an LLM preference profile for a cold-start user."
    )

    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="Cold-start user ID.",
    )

    args = parser.parse_args()

    events, users, movies = load_data()

    likes, dislikes = get_cold_start_user(args.user_id, users)

    print("\nUser self-description:")
    print("\nLikes:")
    print(likes)

    print("\nDislikes:")
    print(dislikes)

    profile = build_preference_profile(likes, dislikes)

    print("\nStructured preference profile:")
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
