# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VIMALA SAHANA W
RegisterNumber: 212223040241 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("Untitled spreadsheet - Sheet1 (1).csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE=',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print('RMSE=',rmse)
```

## Output:

![Screenshot 2024-08-29 190619](https://github.com/user-attachments/assets/0c973004-5e4d-41d3-b34e-2a00c64709f8)

![Screenshot 2024-08-29 190633](https://github.com/user-attachments/assets/7c4e2044-dc81-4de0-a7b4-e55c42d67305)

![Screenshot 2024-08-29 190705](https://github.com/user-attachments/assets/b71c5e40-3058-4b72-8a28-ad7a89dcfe84)


![Screenshot 2024-08-29 190720](https://github.com/user-attachments/assets/001bf0d8-a647-49ff-b517-ae8c8eb57384)


![Screenshot 2024-08-29 190743](https://github.com/user-attachments/assets/f766ea3d-a408-4a14-92bf-529f60256ff8)


![Screenshot 2024-08-29 190756](https://github.com/user-attachments/assets/acb450f9-4819-4b8c-adb5-ccc56904488d)

![Screenshot 2024-08-29 190818](https://github.com/user-attachments/assets/53a67bb3-a1fe-4bdb-b5c1-87a300e68d0e)


![Screenshot 2024-08-29 190836](https://github.com/user-attachments/assets/365a5ee3-dffc-44d8-aa0d-8c718137397b)


![Screenshot 2024-08-29 190846](https://github.com/user-attachments/assets/9824b8e0-0760-44f1-8730-4d059b7e8122)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
