import pandas as pd

movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

movies_df.set_index("movieId", inplace=True)

print(movies_df)