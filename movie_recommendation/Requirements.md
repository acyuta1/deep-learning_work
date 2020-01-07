# Movie Recommendation System [User to Item]

Takes into account the watching pattern of existing users and Recommends movies for a new user.
The model gives importance not only to the preferred genres of a new user, but also the viewing pattern of similar users who have watched similar movies.

# Dependencies Required:
1) Pandas 0.25.1
2) Numpy 1.16.5
3) Keras 2.3.1
4) Tensorflow 2.0.0
5) Pickle

# Dataset: MovieLens (Any)

# IDE's:
Jupyter Notebook and Spyder

# Python .py files of API:
1) server_side.py : Starts flask REST application. On POST request, calls all the required methods to obtain predictions and display them.
2) prediction_helper.py: Has two methods: One to transform raw input to the required input shape. And the other which is straighforward, returns predictions.
3) data_loader.py: Loads all the required data. Also has two methods: One to pad input to the required length and the other to obtain corresponding rows from the dataset once predictions are obtained.
        
