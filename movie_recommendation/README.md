# Movie Recommendation System [User to Item]

Takes into account the watching pattern of existing users and Recommends movies for a new user.
The model gives importance not only to the preferred genres of a new user, but also the viewing pattern of similar users who have watched similar movies.


# Python .py files of API:
1) server.py : Starts flask REST application. On POST request, calls all the required methods to obtain predictions and display them.
2) predict.py: Has two methods: 
        i. prepare_inputs: transforms raw input to the required input shape. 
        ii. make_predictions: returns predictions.
3) data_loader.py: Loads all the required data. Also has two methods: 
        i. pad: pads input to the required length.
        ii. returns corresponding rows from the dataset matching the predicted movieIds.
        
