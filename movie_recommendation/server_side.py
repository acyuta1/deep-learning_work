from data_loader import MODEL, result
from prediction_helper import prepare_inputs, make_prediction
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    if request.method == "POST":

        comment = request.form["comment"]
        movies = [int(x) for x in comment.split(",")]

        movies, g_input, t_input, movie_copy = prepare_inputs(movies)

        print(movies[:10],g_input,t_input)

        preds = make_prediction(movies, g_input, t_input, MODEL)

        most_similar = preds[0].argsort()[-(10 + len(movie_copy)) :][::-1]

        watched_movies, rec_movies = result(movie_copy, most_similar)

        context = {
            "table1": [watched_movies.to_html(classes="data")],
            "title1": watched_movies.columns.values,
            "table2": [rec_movies.to_html(classes="data")],
            "title2": rec_movies.columns.values,
        }

    return render_template("prediction.html", context=context)


if __name__ == "__main__":
    global MODEL
    MODEL = MODEL
    app.run(threaded=False)
