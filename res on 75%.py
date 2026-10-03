import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import skorch
from kagglesdk.competitions.types import submission_status
from narwhals import DataFrame
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV


def maee(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

X_train=df_train.drop(['Survived','Name','Cabin','Ticket','PassengerId'],axis=1)
y_train=df_train['Survived']

X_test=df_test.drop(['Name','Cabin','Ticket','PassengerId'],axis=1)


#привод данных к типу int
X_test.loc[152,'Fare']=0.0
#для train выборки
for idx,val in X_train['Age'].items():
    if pd.isna(val):
        X_train.loc[idx,'Age']=0.0
X_train['Sex']=X_train['Sex'].astype(object)
for idx,val in X_train['Sex'].items():
    if val=='male':
        X_train.loc[idx,'Sex']=2
    else:
        X_train.loc[idx, 'Sex'] = 1
X_train['Embarked'] = X_train['Embarked'].astype(object)
for idx, val in X_train['Embarked'].items():
    if val == 'C':
        X_train.loc[idx, 'Embarked'] = 1
    elif val == 'Q':
        X_train.loc[idx, 'Embarked'] = 2
    elif val == 'S':
        X_train.loc[idx, 'Embarked'] = 3
    else:
        X_train.loc[idx, 'Embarked'] = 0  # NaN case

#для test выборки
for idx,val in X_test['Age'].items():
    if pd.isna(val):
        X_test.loc[idx,'Age']=0.0
X_test['Sex']=X_test['Sex'].astype(object)
for idx,val in X_test['Sex'].items():
    if val=='male':
        X_test.loc[idx,'Sex']=2
    else:
        X_test.loc[idx, 'Sex'] = 1
X_test['Embarked'] = X_test['Embarked'].astype(object)
for idx, val in X_test['Embarked'].items():
    if val == 'C':
        X_test.loc[idx, 'Embarked'] = 1
    elif val == 'Q':
        X_test.loc[idx, 'Embarked'] = 2
    elif val == 'S':
        X_test.loc[idx, 'Embarked'] = 3
    else:
        X_test.loc[idx, 'Embarked'] = 0



#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#print(X_test)
#sys.exit(0)


model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=8

)

model.fit(X_train,y_train)
ypred=model.predict(X_train)
print(maee(y_train,ypred))
prediction=model.predict(X_test)

submission=pd.DataFrame({
    'PassengerId' : df_test['PassengerId'],
    'Survived' : prediction

})

submission.to_csv('submission.csv',index=False)
