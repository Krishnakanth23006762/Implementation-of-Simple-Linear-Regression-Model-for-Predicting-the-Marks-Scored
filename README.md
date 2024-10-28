# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### 1.Import necessary libraries for data handling, visualization, and machine learning.
### 2.Read the student_scores.csv file and display initial data insights.
### 3.Separate features (x) and target (y).
### 4.Split data into training and test sets.
### 5.Fit a linear regression model on the training data.
### 6.Predict on test data and plot both training and test data with regression lines.
### 7.Calculate and print MSE, MAE, and RMSE for model performance.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: B krishnakanth
RegisterNumber:  212223230109
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

df.head()

![dfhead](https://user-images.githubusercontent.com/119393424/229978451-2b6bdc4f-522e-473e-ae2f-84ec824344c5.png)

df.tail()

![dftail](https://user-images.githubusercontent.com/119393424/229978854-6af7d9e9-537f-4820-a10b-ab537f3d0683.png)

Array value of X

![xvalue](https://user-images.githubusercontent.com/119393424/229978918-707c006d-0a30-4833-bf77-edd37e8849bb.png)

Array value of Y

![yvalue](https://user-images.githubusercontent.com/119393424/229978994-b0d2c87c-bef9-4efe-bba2-0bc57d292d20.png)

Values of Y prediction

![ypred](https://user-images.githubusercontent.com/119393424/229979053-f32194cb-7ed4-4326-8a39-fe8186079b63.png)

Array values of Y test

![ytest](https://user-images.githubusercontent.com/119393424/229979114-3667c4b7-7610-4175-9532-5538b83957ac.png)

Training Set Graph

![train](https://user-images.githubusercontent.com/119393424/229979169-ad4db5b6-e238-4d80-ae5b-405638820d35.png)

Test Set Graph

![test](https://user-images.githubusercontent.com/119393424/229979225-ba90853c-7fe0-4fb2-8454-a6a0b921bdc1.png)

Values of MSE, MAE and RMSE


![image](https://github.com/user-attachments/assets/93391cca-6ebc-47dd-a9db-3a53de7af64b)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

