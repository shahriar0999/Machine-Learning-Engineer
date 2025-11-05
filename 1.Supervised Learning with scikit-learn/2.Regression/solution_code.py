# Creating features
import numpy as np

# Create X from the radio column's values
X = sales_df['radio']

# Create y from the sales column's values
y = sales_df['sales']

# Reshape X
X = X.values.reshape(-1, 1)

# Check the shape of the features and targets
print(X.shape, y.shape)

######### Building a linear regression model  ###############

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])