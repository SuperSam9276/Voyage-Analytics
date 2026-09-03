# Flight Price Prediction

## Project Overview

Flight Price Prediction is a machine learning project that predicts the estimated price of a flight based on different flight-related features.

This module is part of our group Machine Learning project. My responsibility in the group project is the Flight Price Prediction module.

## Objective

The main objective is to build a machine learning model that can estimate flight prices based on information such as departure city, destination city, flight type, travel time, distance, agency, and date-related features.

## Features Used

The model uses the following features:

- From
- To
- Flight Type
- Time
- Distance
- Agency
- Year
- Month
- Day
- Day of Week

The target variable is:

- Price

Identifier columns such as travelCode and userCode are not used for prediction.

## Machine Learning Workflow

The project follows these steps:

1. Load the flight dataset.
2. Perform data preprocessing.
3. Extract useful date features such as year, month, day, and day of week.
4. Select relevant features.
5. Split the data into training and testing sets.
6. Preprocess numerical and categorical features.
7. Train a Random Forest Regression model.
8. Evaluate the model.
9. Save the trained model and preprocessing pipeline.
10. Integrate the model with a Flask web application.

## Data Preprocessing

Numerical features are processed separately from categorical features.

### Numerical Features

- Time
- Distance
- Year
- Month
- Day
- Day of Week

### Categorical Features

- From
- To
- Flight Type
- Agency

Categorical features are encoded using OneHotEncoder, while missing values are handled using appropriate imputation techniques.

## Model

A *Random Forest Regressor* is used for flight price prediction.

The trained model and preprocessing pipeline are saved as:

```text
flight_price_model.pkl
flight_price_preprocessor.pkl

## Web Application

The trained flight price prediction model is integrated with a Flask web application.

Users can enter the following flight details:

- Departure City
- Destination City
- Flight Type
- Time
- Distance
- Agency
- Year
- Month
- Day
- Day of Week

The Flask application takes these inputs, applies the saved preprocessing pipeline, and uses the trained Random Forest Regression model to predict the estimated flight price.

The predicted flight price is displayed in *Brazilian Real (R$)*, consistent with the original dataset.

## Project Files

- app.py – Flask application for the flight price prediction webpage
- flight_price_model.pkl – Trained Random Forest Regression model
- flight_price_preprocessor.pkl – Saved preprocessing pipeline
- templates/index.html – HTML webpage for entering flight details
- README_Flight_Price_Prediction.md – Project documentation

## How to Run the Application

1. Open the project folder in VS Code.

2. Open the terminal and navigate to the flight_api folder:

```bash
cd flight_api

3. Run the Flask application
   python app.py

4.The Flask server will start at:

   http://127.0.0.1:5000

5. Open the URL in a web browser.

6. Enter the required flight details and click the prediction button to get the estimated flight price.

## Output

The web application successfully predicts the estimated flight price based on the user-provided flight details.

Example output:

*Predicted Flight Price: R$931.63*

## Conclusion

The Flight Price Prediction model estimates flight prices using different travel-related features such as departure city, destination city, flight type, time, distance, agency, and date-related features.

The trained model is integrated into a Flask web application, allowing users to enter flight details and receive a predicted flight price in Brazilian Real (R$).