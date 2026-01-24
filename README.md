# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PRAVEENA D
RegisterNumber: 212224040248 
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta
data = pd.read_csv("C:\\Users\\admin\\Downloads\\50_Startups.csv", header=None)
print(data.head())
X = (data.iloc[1:, :-2].values)
print(X)
X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
y1_Scaled = scaler.fit_transform(y)
print('Name: PRAVEENA D')
print('Register No.:212224040248')
print(X1_Scaled)
theta = linear_regression(X1_Scaled, y1_Scaled)
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

<img width="819" height="343" alt="image" src="https://github.com/user-attachments/assets/1607da8b-ffac-4e8b-b03c-4b60b3aedca7" />
<img width="411" height="345" alt="image" src="https://github.com/user-attachments/assets/32d67df4-6703-4b37-afb4-b5b1f77dadb9" />
<img width="614" height="551" alt="image" src="https://github.com/user-attachments/assets/368a2143-e531-4269-9f88-4195b88242c9" />
<img width="314" height="594" alt="image" src="https://github.com/user-attachments/assets/5d7dab64-8e28-4f20-9daf-2ce2a2c61acf" />
<img width="305" height="610" alt="image" src="https://github.com/user-attachments/assets/60a800f3-2077-4ad9-b0e2-9271e9d54d42" />
<img width="795" height="611" alt="image" src="https://github.com/user-attachments/assets/3b1178c2-05f7-4b75-b56e-5029b87d77d9" />
<img width="907" height="609" alt="image" src="https://github.com/user-attachments/assets/ca67a999-dc9b-4f7f-8b54-02ca3601a66e" />
<img width="784" height="603" alt="image" src="https://github.com/user-attachments/assets/2d8973c0-62fc-4f8e-ad27-f53774647869" />





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
