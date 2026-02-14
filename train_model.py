# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset
df = pd.read_csv('heart.csv')

# The dataset is already clean – no missing values
# Convert target to binary if necessary (it already is 0/1)
# But let's ensure it's integer
df['target'] = df['target'].astype(int)

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate
accuracy = knn.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save model and scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved.")
