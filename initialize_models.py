from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump
import numpy as np

# Create and save a simple default model
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5,
    random_state=42
)

# Train on some dummy data
X = np.random.rand(100, 6)  # 6 features
y = np.random.randint(0, 2, 100)  # Binary target
model.fit(X, y)

# Save the model
dump(model, 'trading_model.joblib')

# Create and save a scaler
scaler = StandardScaler()
scaler.fit(X)
dump(scaler, 'trading_model_scaler.joblib')

print("Initial model and scaler created successfully!")