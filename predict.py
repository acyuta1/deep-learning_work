import numpy as np
from random import shuffle
from data_loader import DF, pad, INPUT_LENGTH, MOVIES_LIST, GENRES_TOKENS

def make_prediction(*arg):
    """
    Function to predict provided we have the processed inputs. 
    """
    if arg[3]==True:
        movie_pred_indices = []
        g_input = arg[0] 
        model = arg[1]
        movie_copy = arg[2]
        print(movie_copy, g_input)        
        for i in movie_copy:
            preds = model.predict([([g_input[i],])])
            most_similar = preds[0].argsort()[-30:][::-1]
            movie_pred_indices.append(most_similar.tolist())    
        movie_pred_indices = [item for sublist in movie_pred_indices 
                              for item in sublist]
        movie_pred_indices = list(set(movie_pred_indices))
        shuffle(movie_pred_indices)
        movie_pred_indices = [i+1 for i in movie_pred_indices]
        return np.array(movie_pred_indices[:15])           
    else:
        movies = arg[0]
        g_input = arg[1]
        model = arg[2]
        movies = np.asarray(movies).astype('float32')/9724
        preds = model.predict([np.array([movies,]),np.array([g_input,])])
        top_3_sequence = preds[0].argsort()[-3:][::-1]
        movies = [DF.movieId[DF.movieId.isin(MOVIES_LIST[i])].values.tolist() for 
                             i in top_3_sequence]
        movies = [item for sublist in movies for item in sublist]
        return movies
        

def prepare_inputs(*args):
    """
    With only MovieID's as the input to our model, we are required to 
    obtain the required genre's and tag's and bring our raw input to
    the required shape.
    """
    if args[1]==True:        
        movie_input = args[0]
        movie_input = [DF.movieId[DF['title']==i].iloc[0] for i in movie_input]
        movies_copy = movie_input[:]
        movie_input = [i-1 for i in movie_input]
        g_input = GENRES_TOKENS
        return g_input, movie_input, movies_copy
    else:
        movie_input = args[0]
        if len(movie_input)>18:
            movie_input = movie_input[len(movie_input)-18:]
            
        movie_input = [DF.movieId[DF['title']==i].iloc[0] for i in movie_input]
        movies_copy = movie_input[:]
        genres = (DF.genres_class[DF["movieId"].isin(movie_input)].values).tolist()
        genres = pad(genres, INPUT_LENGTH["genre_len"])    
        movie_input = pad(movie_input, INPUT_LENGTH["movie_len"])
        
        return movie_input, genres, movies_copy
        

