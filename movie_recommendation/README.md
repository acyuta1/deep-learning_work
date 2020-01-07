# Movie Recommendation System [User to Item]

Takes into account the watching pattern of existing users and Recommends movies for a new user.
The model gives importance not only to the preferred genres of a new user, but also the viewing pattern of similar users who have watched similar movies.

# Dependencies:

1. pip install pandas
2. pip install keras==2.3.0
3. pip install tensorflow==1.15
3. pip install gunicorn
4. pip install flask

# Python .py files of API:
1) server_side.py : Starts flask REST application. On POST request, calls all the required methods to obtain predictions and display them.
2) prediction_helper.py: Has two methods: One to transform raw input to the required input shape. And the other which is straighforward, returns predictions.
3) data_loader.py: Loads all the required data. Also has two methods: One to pad input to the required length and the other to obtain corresponding rows from the dataset once predictions are obtained.
        
