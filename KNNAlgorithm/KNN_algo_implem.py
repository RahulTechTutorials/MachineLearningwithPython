import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math


os.getcwd()
os.chdir('/Users/rahuljain/Desktop/Python/MachineLearningwithPython/KNNAlgorithm')
os.listdir()
dataset = pd.read_csv('diabetescsv.zip')
dataset.head()

#######Steps for feature engineering#######
##describe will tell you which of the columns have min values as zeros
dataset.describe()
##below code will help you understand if there are any missing values
dataset.isnull().values.any()
remove_zeros = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for column in remove_zeros:
    dataset[column] = dataset[column].replace(0,np.NaN)
    mean = dataset[column].mean(skipna=True)
    dataset[column] = dataset[column].replace(np.NaN,mean)
###By checking again, all the columns must have non zero min values
dataset.describe()

####Splitting the data into train and test#####
##Picking your X and y columns
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
##help(train_test_split)
##splitting the data into test and train
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0, test_size=0.2)
##X_test.info(),X_train.info()

#######Further Steps for feature engineering#######
##StandardScaler standadizes all the values of all the columns
scaledX =  StandardScaler()
X_train = scaledX.fit_transform(X_train)
X_test  = scaledX.transform(X_test) 
X_test

###math.sqrt(len(y_test))-- gives 12, we convert it to 11 for better voting results

####Defining the model####
##help(KNeighborsClassifier)
classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')

###Fit Model
classifier.fit(X_train,y_train)

##Make predictions 
y_pred = classifier.predict(X_test)
y_pred

###Capturing results ####
cm = confusion_matrix(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
score = accuracy_score(y_test,y_pred)

print('---Confusion Matrix---',end='\n')
print(cm)
print('---F1 Score---',end='\n')
print(f1)
print('---Accuracy Score---',end='\n')
print(score)
