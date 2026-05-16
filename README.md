# Ex.No: 6 HOLT WINTERS METHOD
### Date: 


## AIM:
To implement the Holt-Winters Method model on the Apple revenue dataset using Python and forecast future revenue values.
## DATASET
Apple 2009-2005
## SOFTWARE REQUIRED
Google Colab
## ALGORITHM
1. Import the necessary libraries such as pandas, matplotlib, numpy, sklearn, and statsmodels.
2. Load the Apple revenue CSV dataset into a DataFrame.
3. Set the year column as the index and perform initial data exploration.
4. Plot the Apple revenue time series data.
5. Scale the revenue data using MinMaxScaler.
6. Import the required statsmodels libraries for time series analysis.
7. Decompose the revenue data into additive components and plot them.
8. Split the dataset into training and testing data.
9. Create and train the Holt-Winters Exponential Smoothing model.
10. Predict the test dataset values using the trained model.
11. Calculate the Root Mean Squared Error (RMSE) to evaluate the model performance.
12. Calculate the mean and standard deviation of the scaled revenue dataset.
13. Fit the Holt-Winters model to the complete dataset and forecast future revenue values.
14. Plot the original revenue data and future predictions.


## PROGRAM
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Step 1: Load dataset
data = pd.read_csv("Apple 2009-2024.csv")

# Step 2: Set year as index
data.set_index('year', inplace=True)

# Step 3: Convert Revenue column to numeric
data['Revenue (millions)'] = (
    data['Revenue (millions)']
    .replace('[\$,]', '', regex=True)
    .astype(float)
)

# Step 4: Plot original data
plt.figure(figsize=(10,5))
plt.plot(data.index, data['Revenue (millions)'])
plt.title("Apple Revenue Data")
plt.xlabel("Year")
plt.ylabel("Revenue (millions)")
plt.show()

# Step 7: Predict test data
test_predictions = model.forecast(len(test_data))

# Step 8: Plot train, test and predictions
plt.figure(figsize=(10,5))

plt.plot(train_data.index, train_data, label='Train Data')
plt.plot(test_data.index, test_data, label='Test Data')
plt.plot(test_data.index, test_predictions, label='Predictions')

plt.legend()
plt.title("Visual Evaluation")
plt.xlabel("Year")
plt.ylabel("Revenue")
plt.show()

# Step 9: Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))

print("RMSE Value:", rmse)

# Step 10: Create final model
final_model = ExponentialSmoothing(
    data['Revenue (millions)'],
    trend='add',
    seasonal=None
).fit()

# Step 11: Forecast future revenue for next 4 years
future_predictions = final_model.forecast(4)

# Step 12: Plot final predictions
plt.figure(figsize=(10,5))

plt.plot(data.index, data['Revenue (millions)'], label='Original Data')

future_years = [2025, 2026, 2027, 2028]

plt.plot(
    future_years,
    future_predictions,
    label='Future Predictions'
)

plt.legend()
plt.title("Apple Revenue Forecast")
plt.xlabel("Year")
plt.ylabel("Revenue (millions)")
plt.show()
```

## OUTPUT

<img width="675" height="351" alt="image" src="https://github.com/user-attachments/assets/1c346214-220a-4639-9f4c-441ae4925dca" />

<img width="689" height="368" alt="image" src="https://github.com/user-attachments/assets/05c2fa09-4783-4200-baf5-32d7d3d114bb" />

<img width="681" height="365" alt="image" src="https://github.com/user-attachments/assets/81f43644-8abd-4c26-9f68-74e469a48fc1" />

## RESULT:

Thus the program run successfully based on the Holt Winters Method model.
