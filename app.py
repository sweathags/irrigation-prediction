# app.py
from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load the saved model
loaded_model = load("random_forest_model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input data from the form
        CropType = request.form["CropType"]
        CropDays = request.form["CropDays"]
        SoilMoisture = request.form["SoilMoisture"]
        temperature = request.form["temperature"] 
        Humidity = request.form["Humidity"]

        # Create new_data DataFrame
        new_data = pd.DataFrame({
            'CropType': [CropType],
            'CropDays': [CropDays],
            'SoilMoisture': [SoilMoisture],
            'temperature': [temperature],
            'Humidity': [Humidity]
        })

        # Make predictions
        predictions = loaded_model.predict(new_data)
        probability = loaded_model.predict_proba(new_data)[:,1]

        # Render template with predictions
        return render_template("result.html", prediction=predictions[0], probability=probability[0])

    # Render the form
    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
