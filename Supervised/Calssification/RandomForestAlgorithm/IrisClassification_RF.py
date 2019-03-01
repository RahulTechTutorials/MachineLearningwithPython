import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
###setting seed as 0 to have the same random allocation everytime###
np.random.seed(0)

iris = load_iris()
##type(iris)
iris.data
iris.feature_names
iris.target
iris.target_names

##loading the data in the dataframe
df = pd.DataFrame(iris.data,columns = iris.feature_names)
df
iris.target
iris.target_names

####Importing the species (dependent variable) as value. Please note the target is factorized hence need to import from codes and target names
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)
df.head()

##spliting the data into train and test
df['is_train'] = np.random.uniform(0,1,len(df)) <= 0.75
df

train, test = df[df['is_train'] == True],df[df['is_train'] == False]
train.count(),test.count()

###Factorize our target y
type(pd.factorize(train['species']))
y  = pd.factorize(train['species'])[0]
type(y)
y

##Prediction by Model
##help(RandomForestClassifier)
clf = RandomForestClassifier(n_estimators=100,criterion='gini',max_features='auto',n_jobs=2,random_state=0)
#help(clf.fit)
train[iris.feature_names]
y

y_test = test['species']
clf.fit(train[iris.feature_names],y)
y_pred = clf.predict(test[iris.feature_names])
##clf.predict_proba(test[iris.feature_names])
##y_test = test['species']
y_pred
y_pred_names = iris.target_names[y_pred]
type(y_pred_names)
y_pred_names.shape
y_test.head(10),y_pred_names[0:10]

len(y_test),len(y_pred)
####Analyzing the results in a cross tab format
cm = pd.crosstab(y_test,y_pred_names,rownames= ['Actual Species'],colnames=['Predicted Species'])

accuracy_score = accuracy_score(y_pred_names,y_test)
#help(f1_score)
f1_score = f1_score(y_test,y_pred_names,average='micro')
precision_score = precision_score(y_test, y_pred_names, pos_label='positive',average='micro')
recall_score = recall_score(y_test, y_pred_names, pos_label='positive',average='micro')

####Displaying the results ####
print('Confusion Matrix','\n',cm,'\n\n','F1 Score: ',f1_score,'\n\n','Accuracy Score','\n',accuracy_score,'\n')
print('Precision_score','\n',precision_score,'\n\n','Recall_score: ',recall_score,'\n')
