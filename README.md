# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:STARBIYA S 
RegisterNumber:  212223040208
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
![Screenshot 2024-08-29 190619](https://github.com/user-attachments/assets/c3936fa8-bee1-44b0-9524-26bb5aa428b2)

![Screenshot 2024-08-29 190633](https://github.com/user-attachments/assets/509e7957-90fc-4c48-ab54-44c2c25040ed)

![Screenshot 2024-08-29 190705](https://github.com/user-attachments/assets/11fbb9bd-2bfd-4728-a331-dee49ef1fe61)

![Screenshot 2024-08-29 190720](https://github.com/user-attachments/assets/5b673e5c-6a6b-422f-b996-7edfea66fa44)

![Screenshot 2024-08-29 190743](https://github.com/user-attachments/assets/1bc5b902-18a7-4912-be86-c69054f390c0)

![Screenshot 2024-08-29 190756](https://github.com/user-attachments/assets/a7eb00cd-0b09-427a-923d-f944cca60806)

![Screenshot 2024-08-29 190818](https://github.com/user-attachments/assets/a79ededc-aad1-41ee-ac83-f39ca3b4b7c4)

![Screenshot 2024-08-29 190836](https://github.com/user-attachments/assets/a2eccc15-f450-4dce-8eca-66450774d2cd)

![Screenshot 2024-08-29 190846](https://github.com/user-attachments/assets/8f1b1a20-7d3b-465c-a897-bd90fafef6cd)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
