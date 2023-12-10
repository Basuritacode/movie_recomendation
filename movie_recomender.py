import pandas as pd
import numpy

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


# users are placed on a plane and the distance determines how similar they are
def get_user_ratings(user_id):
    user_ratings = ratings_df.query(f" userId == {user_id} ")
    return user_ratings[["movieId", "rating"]].set_index("movieId")


def get_users_distance(user_id_1,user_id_2):
    """
    Returns a list with the users id and their respective distance based on same movies rated
    """
    user_1 = get_user_ratings(user_id_1)
    user_2 = get_user_ratings(user_id_2)

    users_compared = user_1.join(user_2, lsuffix="_u1", rsuffix="_u2").dropna()

    # calculate the distance using a library
    user_1_filtered = users_compared["rating_u1"]
    user_2_filtered = users_compared["rating_u2"]
    distance  = numpy.linalg.norm(user_1_filtered - user_2_filtered)

    return [user_id_1, user_id_2, distance] 


def get_all_distances(user_id):
    """
    calculate the relative distance from 'user_id' to all other users in a Dataframe
    """
    users = ratings_df["userId"].unique()
    other_users = users[users != user_id]

    distances = [ get_users_distance(user_id, other_id) for other_id in other_users ]

    return pd.DataFrame(distances, columns=["userId", "otherId", "distance"])


def get_top_matches(user_id):
    """
    Sorts and returns top matches between similiar users
    """
    distance_to_user = get_all_distances(user_id)
    return  distance_to_user.sort_values("distance").set_index("userId")
    


print(get_top_matches(9))