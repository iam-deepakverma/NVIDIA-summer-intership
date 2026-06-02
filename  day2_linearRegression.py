import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Kaggle dataset
data = pd.read_csv('train.csv')
df = pd.DataFrame(data)
# Select feature and target
X = df[['GrLivArea']]
y = df['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Predict a new house
new_house = [[2000]]
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])


