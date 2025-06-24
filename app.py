from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form["news"]
        if user_input.strip() == "":
            prediction = "Please enter some news text."
        else:
            # Preprocess and predict
            input_vector = vectorizer.transform([user_input])
            result = model.predict(input_vector)
            prediction = f"This news is **{result[0]}**."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
