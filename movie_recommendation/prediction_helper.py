import numpy as np
from data_loader import DF, GENRES_MAP, TAG_MAP, pad, INPUT_LENGTH


def make_prediction(movies, g_input, t_input, model):
    """
    Function to predict provided we have the processed inputs. 
    """
    return model.predict(
        [np.array([movies,]), np.array([g_input,]), np.array([t_input,])]
    )


def prepare_inputs(movie_input):
    """
    With only MovieID's as the input to our model, we are required to 
    obtain the required genre's and tag's and bring our raw input to
    the required shape.
    """
    movies_copy = movie_input[:]

    genres = (DF.genres[DF["movieId"].isin(movie_input)].values).tolist()
    genres = [word for line in genres for word in line.split()]

    tags = (DF.tag[DF["movieId"].isin(movie_input)].values).tolist()
    tags = [word for line in tags for word in line.split()]

    genres = [
        list(GENRES_MAP.keys())[list(GENRES_MAP.values()).index(i)] for i in genres
    ]
    genres = list(set(genres))
    genres = pad(genres, INPUT_LENGTH["genre_len"])
    
    tags = [
        list(TAG_MAP.keys())[list(TAG_MAP.values()).index(i)]
        for i in tags
        if i not in ("em", "cs", "se")
    ]
    tags = list(set(tags))
    if len(tags) > 100:
        tags = tags[0:100]
    else:
        tags = pad(tags, INPUT_LENGTH["tag_len"])
        
    movie_input = pad(movie_input, INPUT_LENGTH["movie_len"])
    
    return movie_input, genres, tags, movies_copy
