# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 2: Load your Salary.csv
df = pd.read_csv('Salary.csv')

# Step 3: Encode categorical columns if present (like "Position")
if df['Position'].dtype == 'object':
    le = LabelEncoder()
    df['Position'] = le.fit_transform(df['Position'])

# Step 4: Define features (X) and target (y)
X = df[['Level']]  # 'Level' as input feature
y = df['Salary']   # 'Salary' as target/output

# Step 5: Fit the Decision Tree Regressor
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X, y)

# Step 6: Make predictions (for the training set -- as the dataset is very small)
y_pred = reg.predict(X)

# Step 7: Output predictions and, optionally, plot the results
print("Actual salaries:", list(y))
print("Predicted salaries:", list(y_pred))

# Plotting
plt.scatter(X, y, color='red', label='Actual')
plt.plot(X, y_pred, color='blue', label='Predicted (Decision Tree)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Decision Tree Regression for Employee Salary')
plt.legend()
plt.show()

```

## Output:


<img width="913" height="127" alt="image" src="https://github.com/user-attachments/assets/340bc112-ba28-4849-a69b-58c964215aef" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
