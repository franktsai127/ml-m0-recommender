# ML in Production - Milestone 0

Individual movie recommendation model for CMU ML in Production.

The system uses two recommendation paths:

- Existing users with rating history: item-based collaborative filtering.
- Cold-start users with no interaction history: LLM-assisted content-based recommendation using the user's self-described likes and dislikes.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Download the three Milestone 0 datasets:

```bash
mkdir -p data

curl -fL https://github.com/mlip-cmu-online/public-data/raw/refs/heads/main/m0/data/events.csv.gz -o data/events.csv.gz

curl -fL https://github.com/mlip-cmu-online/public-data/raw/refs/heads/main/m0/data/users.csv.gz -o data/users.csv.gz

curl -fL https://github.com/mlip-cmu-online/public-data/raw/refs/heads/main/m0/data/movies.csv.gz -o data/movies.csv.gz
```

The files do not need to be extracted. Pandas reads the gzip-compressed CSV files directly.

## OpenAI API Setup

Cold-start recommendations use an LLM to convert user self-described likes and dislikes into a structured preference profile.

Create a `.env` file in the repository root:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=your_model_name_here
```

The `.env` file is excluded from Git and should not be committed.

## Training

Train the existing-user recommendation model:

```bash
python src/train.py
```

This creates the model artifacts under `models/`.

The model uses explicit 1-10 movie ratings to construct an item-item cosine similarity matrix.

## Recommendation

Generate recommendations for any user ID:

```bash
python src/recommend.py --user-id USER_ID
```

Return a specific number of recommendations:

```bash
python src/recommend.py --user-id USER_ID --top-k 5
```

The recommendation path is selected automatically:

- Users with rating history use item-based collaborative filtering.
- Users without rating history use the LLM-assisted cold-start recommender.

Example existing user:

```bash
python src/recommend.py --user-id 1 --top-k 5
```

Example cold-start user:

```bash
python src/recommend.py --user-id 1001 --top-k 5
```

## Project Structure

```text
.
├── data/
├── models/
├── src/
│   ├── explore.py
│   ├── preprocess.py
│   ├── train.py
│   ├── recommend.py
│   ├── cold_start.py
│   └── cold_start_recommend.py
├── .gitignore
├── README.md
├── requirements.txt
└── m0_report.md
```