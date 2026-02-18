import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[['feature']]
y = data['target']

# Train model
model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("R2 Score:", r2_score(y, predictions))
