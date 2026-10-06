from flask import Flask, render_template, request
import joblib

# Create Flask application
app = Flask(__name__)

# Load the trained ML model
model = joblib.load("house_price_model.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    # Get input from the user
    area = float(request.form["area"])
    bedrooms = float(request.form["bedrooms"])

    # Make prediction
    prediction = model.predict([[area, bedrooms]])

    # Return prediction to webpage
    return render_template(
        "index.html",
        prediction=round(prediction[0], 2)
    )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)
    