import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[['feature']]
y = data['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

predictions = model.predict(X_scaled)

print("R2 Score:", r2_score(y, predictions))
