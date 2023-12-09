import pandas as pd

# Parameters
HEAD = 10
MINIMUN_RATINGS = 100

# Datasets
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# Inplace means that it overrides the variable
movies_df.set_index("movieId", inplace=True)

# get ratings counts foreach movie: MovieId, Ratings counts
movie_rating_counts = ratings_df["movieId"].value_counts()
movies_df["ratingsCount"] = movie_rating_counts

# Sort by top rated
movies_df.sort_values("ratingsCount", ascending=False).head(HEAD)

# Group by average ratings
average_ratings = ratings_df.groupby("movieId").mean()["rating"]
movies_df["averageRating"] = average_ratings

# filtering on a minimun amount of ratings
ratings_treshold: pd.DataFrame = movies_df.query(f" ratingsCount >= {MINIMUN_RATINGS} ").sort_values("ratingsCount")

print(ratings_treshold)