import numpy as np
import joblib
from django.shortcuts import render
from .forms import PredictionForm  # we'll create this

# Load model and scaler once at startup
model = joblib.load('predictor/ml_models/knn_model.pkl')
scaler = joblib.load('predictor/ml_models/scaler.pkl')

def index(request):
    form = PredictionForm()
    return render(request, 'predictor/index.html', {'form': form})

def result(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract features in the correct order
            features = [
                form.cleaned_data['age'],
                form.cleaned_data['sex'],
                form.cleaned_data['cp'],
                form.cleaned_data['trestbps'],
                form.cleaned_data['chol'],
                form.cleaned_data['fbs'],
                form.cleaned_data['restecg'],
                form.cleaned_data['thalach'],
                form.cleaned_data['exang'],
                form.cleaned_data['oldpeak'],
                form.cleaned_data['slope'],
                form.cleaned_data['ca'],
                form.cleaned_data['thal'],
            ]
            # Convert to numpy array and scale
            features_array = np.array([features])
            features_scaled = scaler.transform(features_array)
            # Predict
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]

            context = {
                'prediction': prediction,
                'probability_no': probability[0],
                'probability_yes': probability[1],
            }
            return render(request, 'predictor/result.html', context)
    # If GET or invalid, go back to form
    return index(request)
