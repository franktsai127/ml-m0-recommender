import pandas as pd

EVENTS_PATH = "data/events.csv.gz"
USERS_PATH = "data/users.csv.gz"
MOVIES_PATH = "data/movies.csv.gz"


def inspect_dataframe(name, df):
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum())

    print()


def main():
    events = pd.read_csv(EVENTS_PATH)
    users = pd.read_csv(USERS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    inspect_dataframe("EVENTS", events)
    inspect_dataframe("USERS", users)
    inspect_dataframe("MOVIES", movies)

    print("\nEVENTS unique values by column:")
    for column in events.columns:
        if events[column].nunique() < 20:
            print(events[column].value_counts(dropna=False))

    # Cold-start users
    cold_start_events = events[events["event_type"] == "account_created"]
    cold_start_ids = set(cold_start_events["user_id"])

    print("\n=== Cold Start Checks ===")
    print("Unique cold-start users:", len(cold_start_ids))

    cold_start_users = users[users["user_id"].isin(cold_start_ids)]

    print("Cold-start users found in users.csv:", len(cold_start_users))
    print(
        "Cold-start users missing likes:",
        cold_start_users["self_description_likes"].isna().sum(),
    )
    print(
        "Cold-start users missing dislikes:",
        cold_start_users["self_description_dislikes"].isna().sum(),
    )

    # Make sure cold-start users have no watch/rating history
    interaction_ids = set(
        events.loc[events["event_type"].isin(["watch", "rating"]), "user_id"]
    )

    print(
        "Cold-start users with interaction history:",
        len(cold_start_ids & interaction_ids),
    )

    # Integrity checks
    print("\n=== Integrity Checks ===")

    print(
        "Event users missing from users.csv:",
        len(set(events["user_id"]) - set(users["user_id"])),
    )

    event_movies = set(events["movie_id"].dropna())

    print(
        "Event movies missing from movies.csv:",
        len(event_movies - set(movies["movie_id"])),
    )


if __name__ == "__main__":
    main()
