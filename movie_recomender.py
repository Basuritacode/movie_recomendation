import pandas as pd
import numpy
import random

# Parameters
HEAD = 10
MINIMUN_RATINGS = 100

# Datasets
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")
ratings_df.drop(columns=['timestamp'], inplace=True)

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
    return  distance_to_user.sort_values("distance").set_index("otherId")
    

def recommend_movie(user_id):
    """
    makes a movie recomendation based on the top rating of the most similar user
    """
    user_ratings = get_user_ratings(user_id)
    top_match = get_top_matches(user_id).iloc[0]

    match_ratings = get_user_ratings(top_match.name)
    unwatched_movies = match_ratings.drop(user_ratings.index, errors="ignore")
    unwatched_movies = unwatched_movies.sort_values("rating", ascending=False)
    
    return unwatched_movies.join(movies_df)


# K-Nearest Neighbors (KNN) is a supervised machine learning algorithm
NEIGHBORS = 5

def get_knn(user_id, k=NEIGHBORS):
    return get_top_matches(user_id).head(k)


def recommend_movie_with_knn(user_id, k=NEIGHBORS):
    """
    makes a movie recomendation based on the k nearest neighbors ratings 
    """
    user_ratings = get_user_ratings(user_id)
    top_matches = get_knn(k)
    ratings_by_index = ratings_df.set_index("userId")
    top_match_ratings = ratings_by_index.loc[top_matches.index]

    ratings_avg = top_match_ratings.groupby("movieId").mean()[["rating"]]
    unwatched_movies = ratings_avg.drop(user_ratings.index, errors="ignore")
    unwatched_movies = unwatched_movies.sort_values("rating", ascending=False)

    return unwatched_movies.join(movies_df)


# Training the Model
MOVIES = 15
MIN_MOVIEID = 1
MAX_MOVIEID = movies_df.shape[0]
MIN_RATING = 0.0
MAX_RATING = 5.0

watched_movies = []
test_ratings = []

for i in range(0, MOVIES):
    random_movie = random.randint(MIN_MOVIEID, MAX_MOVIEID)
    watched_movies.append(random_movie)

    random_rating = random.randint(MIN_RATING, MAX_RATING)
    test_ratings.append(random_rating)

user_data = [list(index) for index in zip(watched_movies, test_ratings)]

def create_new_user(user_data):
    new_user_id = ratings_df["userId"].max() + 1
    new_ratings = pd.DataFrame(user_data, columns=["movieId", "rating"])
    new_ratings["userId"] = new_user_id

    return pd.concat([ratings_df, new_ratings])



print(recommend_movie_with_knn(611, 4))

# print(recommend_movie_with_knn(9))