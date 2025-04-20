# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the Program.
2.Import the necessary packages.
3.Read the given csv file and display the few contents of the data.
4.Assign the features for x and y respectively.
5.Split the x and y sets into train and test sets.
6.Convert the Alphabetical data to numeric using CountVectorizer.
7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8.Find the accuracy of the model.
9.End the program.
```

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: G.SANJAY
RegisterNumber:  212224230243
*/
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### DATA HEAD:
![image](https://github.com/user-attachments/assets/731c5f64-6139-4e41-b10e-9e3917c003a1)
### DATA INFO:
![image](https://github.com/user-attachments/assets/443ae43d-3061-4c2a-a211-932b9550cd9b)
### NULL:
![image](https://github.com/user-attachments/assets/a4279392-b7ff-4ce5-9ac3-07ea337137fa)
### PREDICTION:
![image](https://github.com/user-attachments/assets/b1b9a421-1a65-406d-b527-3a631851b092)
### ACCURACY:
![image](https://github.com/user-attachments/assets/dc0b5a2d-6bc6-4d07-9478-70df5c2ad632)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
