from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved objects (models, scalers, and encoders)
saved_objects = joblib.load('models/all_models_scalers_encoders.pkl')
models = saved_objects['models']
scalers = saved_objects['scalers']
label_encoders = saved_objects['label_encoders']

# Common Features and Seasonal Features
common_features = ['Ph', 'K', 'P', 'N', 'S', 'Zn', 'Soilcolor']
seasonal_features = {
    'Winter': ['QV2M-W', 'T2M_MAX-W', 'T2M_MIN-W', 'PRECTOTCORR-W'],
    'Spring': ['QV2M-Sp', 'T2M_MAX-Sp', 'T2M_MIN-Sp', 'PRECTOTCORR-Sp'],
    'Summer': ['QV2M-Su', 'T2M_MAX-Su', 'T2M_MIN-Su', 'PRECTOTCORR-Su'],
    'Autumn': ['QV2M-Au', 'T2M_MAX-Au', 'T2M_MIN-Au', 'PRECTOTCORR-Au']
}

@app.route('/', methods=['GET', 'POST'])
def index():
    print("Received request!")  # Debug print

    # Define seasonal features here (you already have this in your code)
    seasonal_features = {
        'Winter': ['QV2M-W', 'T2M_MAX-W', 'T2M_MIN-W', 'PRECTOTCORR-W'],
        'Spring': ['QV2M-Sp', 'T2M_MAX-Sp', 'T2M_MIN-Sp', 'PRECTOTCORR-Sp'],
        'Summer': ['QV2M-Su', 'T2M_MAX-Su', 'T2M_MIN-Su', 'PRECTOTCORR-Su'],
        'Autumn': ['QV2M-Au', 'T2M_MAX-Au', 'T2M_MIN-Au', 'PRECTOTCORR-Au']
    }

    if request.method == 'POST':
        # Handle form submission and predictions
        common_inputs = {}
        for feature in common_features:
            if feature == 'Soilcolor':
                value = request.form.get(feature)
                print(f"Value for {feature}: {value}")  # Print the input value
                value = label_encoders['Soilcolor'].transform([value])[0]
            else:
                value = float(request.form.get(feature))
            common_inputs[feature] = value
        
        predictions = {}
        
        # Make predictions for each season
        for season, features in seasonal_features.items():
            seasonal_inputs = common_inputs.copy()
            for feature in features:
                value = float(request.form.get(feature))
                seasonal_inputs[feature] = value
            
            input_data = np.array(list(seasonal_inputs.values())).reshape(1, -1)
            input_data_scaled = scalers[season].transform(input_data)
            predicted_crop = models[season].predict(input_data_scaled)
            crop_name = label_encoders['label'].inverse_transform([predicted_crop[0]])[0]
            predictions[season] = crop_name
        
        return render_template('index.html', predictions=predictions, seasonal_features=seasonal_features)

    return render_template('index.html', predictions=None, seasonal_features=seasonal_features)



if __name__ == '__main__':
    app.run(port="2000",debug=True)
