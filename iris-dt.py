import pandas as pd
import numpy as np
import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

n_estimators=100
max_depth=10

with mlflow.start_run():
    rfc=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)

    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_param('max_depth',max_depth)

    rfc.fit(X_train,y_train)

    y_pred=rfc.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred,average="weighted")
    f1=f1_score(y_test,y_pred,average="weighted")
    precision=precision_score(y_test,y_pred,average="weighted")

    mlflow.log_metric('accuracy_score',acc)
    mlflow.log_metric('recall_score',recall)
    mlflow.log_metric('f1_score',f1)
    mlflow.log_metric('precision_score',precision)

    