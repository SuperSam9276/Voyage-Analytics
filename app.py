from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and preprocessor
model = joblib.load("flight_price_model.pkl")
preprocessor = joblib.load("flight_price_preprocessor.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    
    try:
        # Get input values from the HTML form
        from_city = request.form.get("from")
        to_city = request.form.get("to")
        flight_Type = request.form.get("flightType")
        time = request.form.get("time")
        distance = request.form.get("distance")
        agency = request.form.get("agency")
        year = request.form.get("year")
        month = request.form.get("month")
        day = request.form.get("day")
        day_of_week = request.form.get("day_of_week")
        

        # Create input dictionary
        data = {
            "from": from_city,
            "to": to_city,
            "flightType": flight_Type,
            "time": float(time),
            "distance": float(distance),
            "agency": agency,
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "day_of_week": int(day_of_week)
        }

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Apply preprocessing
        X = preprocessor.transform(df)

        # Predict flight price
        prediction = model.predict(X)

        # Show prediction
        return f"""
        <h1>Flight Price Prediction</h1>
        <h2>Predicted Flight Price: R$ {float(prediction[0]):.2f}</h2>
        <br>
        <a href="/">Go Back</a>
        """

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)