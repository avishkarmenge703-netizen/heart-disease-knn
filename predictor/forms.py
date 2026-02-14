from django import forms

class PredictionForm(forms.Form):
    age = forms.IntegerField(label='Age', min_value=20, max_value=100, initial=50)
    sex = forms.ChoiceField(label='Sex', choices=[(0, 'Female'), (1, 'Male')])
    cp = forms.ChoiceField(label='Chest Pain Type', choices=[
        (0, 'Typical angina'), (1, 'Atypical angina'), (2, 'Non-anginal pain'), (3, 'Asymptomatic')
    ])
    trestbps = forms.IntegerField(label='Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, initial=120)
    chol = forms.IntegerField(label='Cholesterol (mg/dl)', min_value=100, max_value=400, initial=200)
    fbs = forms.ChoiceField(label='Fasting Blood Sugar > 120 mg/dl', choices=[(0, 'False'), (1, 'True')])
    restecg = forms.ChoiceField(label='Resting ECG Results', choices=[
        (0, 'Normal'), (1, 'ST-T wave abnormality'), (2, 'Left ventricular hypertrophy')
    ])
    thalach = forms.IntegerField(label='Max Heart Rate Achieved', min_value=60, max_value=220, initial=150)
    exang = forms.ChoiceField(label='Exercise Induced Angina', choices=[(0, 'No'), (1, 'Yes')])
    oldpeak = forms.FloatField(label='ST Depression', min_value=0.0, max_value=6.0, initial=1.0)
    slope = forms.ChoiceField(label='Slope of ST Segment', choices=[
        (0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')
    ])
    ca = forms.ChoiceField(label='Number of Major Vessels (0-3)', choices=[(0,0),(1,1),(2,2),(3,3)])
    thal = forms.ChoiceField(label='Thalassemia', choices=[
        (1, 'Normal'), (2, 'Fixed defect'), (3, 'Reversible defect')
    ])
