import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load training data
train_set = pd.read_csv('train.csv')

# Handling missing data
train_set.dropna(inplace=True)

# Prepare features and target variable for training
X_train = train_set.iloc[:, :-1].values
y_train = train_set.iloc[:, -1].values

# Load testing data
test_set = pd.read_csv('test.csv')

# Prepare features and target variable for testing
X_test = test_set.iloc[:, :-1].values
y_test = test_set.iloc[:, -1].values

# Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Print predicted vs actual values (optional)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# Plotting the results
plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.scatter(X_test, y_pred, color='blue', label='Predicted', alpha=0.5)
plt.xlabel('Linear Predicted')
plt.ylabel('Target')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.axis('scaled')

# Calculate and print the R-squared score
r2 = r2_score(y_test, y_pred)
print(f'R-squared score: {r2:.4f}')

plt.show()







'''This code is a simple implementation of a linear regression model using Python's libraries such as `pandas`, `numpy`, `matplotlib`, and `scikit-learn`. Here's a breakdown of what the code does and potential use cases:

Code Breakdown

1. Import Libraries:
   - `numpy`: For numerical operations.
   - `pandas`: For data manipulation and analysis, particularly with structured data.
   - `matplotlib.pyplot`: For data visualization.
   - `sklearn.linear_model`: To access linear regression functionality.
   - `sklearn.metrics`: To evaluate the model's performance.

2. Load Training Data:
   - Reads a CSV file named `train.csv` into a DataFrame.
   - Drops any rows with missing values to ensure clean data for training.

3. Prepare Features and Target:
   - Splits the DataFrame into feature variables (`X_train`) and the target variable (`y_train`), using all columns except the last one for features and the last column for the target.

4. Load Testing Data:
   - Reads a CSV file named `test.csv` into another DataFrame.
   - Similar to the training set, it separates features (`X_test`) and the target variable (`y_test`).

5. Train the Model:
   - Initializes a `LinearRegression` object and fits it to the training data (`X_train` and `y_train`).

6. Make Predictions:
   - Uses the trained model to predict the target variable for the test set (`y_pred`).

7. Visualization:
   - Plots a scatter plot of the actual test values (`y_test`) in red and the predicted values (`y_pred`) in blue.
   - The plot includes axis labels, a title, and a legend for clarity.

8. Model Evaluation:
   - Calculates the R-squared score, which indicates how well the model explains the variability of the target variable in relation to the features. This score is printed out.

Potential Use Cases

1. Predictive Analytics: 
   - Can be used in various fields such as finance (predicting stock prices), marketing (forecasting sales), and real estate (estimating property values).

2. Data Analysis:
   - Helps in analyzing relationships between variables in a dataset. For instance, one might want to see how different features affect a certain outcome.

3. Machine Learning Education:
   - This code serves as a practical example for those learning about machine learning concepts, specifically linear regression.

4. Business Intelligence:
   - Organizations can use it to build models that predict trends or outcomes based on historical data, aiding decision-making processes.

5. Scientific Research:
   - In fields like biology or social sciences, researchers can apply linear regression to understand correlations and influences among measured variables.

Limitations
- The model assumes a linear relationship between features and the target variable, which may not always be valid.
- Handling of categorical variables and more complex datasets would require additional preprocessing.
- It lacks feature scaling, which might be necessary for certain datasets to improve model performance.'''