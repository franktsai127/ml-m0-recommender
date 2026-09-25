# Milestone 0 Report

## Learning

I use the provided event, user, and movie datasets. The event data contains watch and rating events for 1,000 users with interaction history, together with account-created events for 50 cold-start users. Ratings range from 1 to 10. The user data contains demographic information and free-form descriptions of movies users like and dislike. The movie data contains metadata including title, genres, overview, popularity, ratings, external IDs, and license cost.

For users with interaction history, I use item-based collaborative filtering. I construct a user-movie rating matrix from explicit rating events and compute cosine similarity between movie rating vectors. To generate recommendations, I combine the similarity between candidate movies and movies previously rated by the user with the user's centered rating values. Movies that the user has already watched are removed from the candidate set. I selected this approach because the dataset contains explicit user-movie ratings and item-based collaborative filtering provides a simple and reproducible way to learn recommendations from these interactions.

The training implementation is in `src/train.py`. Data loading and preprocessing are implemented in `src/preprocess.py`, and existing-user recommendation scoring is implemented in `src/recommend.py`.

The 50 cold-start users have no watch or rating history, so interaction-based collaborative filtering cannot infer their preferences. Each of these users does, however, provide self-described movie likes and dislikes. I use an LLM to transform these free-form descriptions into a structured preference profile containing liked genres, keywords, and movies, together with disliked genres, keywords, and movies. This implementation is in `src/cold_start.py`.

For cold-start recommendation, movie title, genre, and overview metadata are represented using TF-IDF. Cosine similarity measures how closely each movie matches the user's positive preference profile. Similarity to negative preferences and explicit disliked genres are used as penalties, and movies explicitly mentioned as liked or disliked are excluded from the recommendation candidates. The cold-start ranking implementation is in `src/cold_start_recommend.py`.

## Running Your Model

Install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the three provided Milestone 0 datasets into the `data/` directory using the commands in `README.md`.

For cold-start recommendations, create a `.env` file containing a valid OpenAI API key and model name:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=your_model_name_here
```

Train the collaborative-filtering model:

```bash
python src/train.py
```

Generate recommendations for a user:

```bash
python src/recommend.py --user-id USER_ID
```

Optionally specify the number of recommendations:

```bash
python src/recommend.py --user-id USER_ID --top-k 5
```

The program automatically uses collaborative filtering for users with historical ratings and the LLM-assisted cold-start approach for users without interaction history.