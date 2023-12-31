import pandas

"""
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	          user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC  

u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.

u.user     -- Demographic information about the users; this is a tab
              separated list of
              user id | age | gender | occupation | zip code
              The user ids are the ones used in the u.data data set.
"""

# load the data
data_columns = ["user_id", "item_id", "rating", "timestamp"]
user_columns = ["user_id", "age", "gender", "occupation", "zip_code"]
item_columns = [
	"movie_id"
	,"movie_title"
	,"release_date"
	,"video_release_date"
	,"imdb_url"
	,"unknown"
	,"action"
	,"adventure"
	,"animation"
	,"childrens"
	,"comedy"
	,"crime"
	,"documentary"
	,"drama"
	,"fantasy"
	,"film_noir"
	,"horror"
	,"musical"
	,"mystery"
	,"romance"
	,"sci_fi"
	,"thriller"
	,"war"
	,"western"
]

data_df = pandas.read_csv("u.data", names = data_columns, sep = "\t", encoding = "ISO-8859-1")
item_df = pandas.read_csv("u.item", names = item_columns, sep = "|", encoding = "ISO-8859-1")
user_df = pandas.read_csv("u.user", names = user_columns, sep = "|", encoding = "ISO-8859-1")



print(user_df) 