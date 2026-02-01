import pandas as pd
import numpy as np
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
import dagshub

dagshub.init(repo_owner='jay-kanakia', repo_name='mlflow-remote-tracking-dagshub-demo', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/mlflow-remote-tracking-dagshub-demo.mlflow')

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

n_estimators=100
max_depth=10

mlflow.set_experiment('iris-dt')
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
    cm=confusion_matrix(y_test,y_pred)

    mlflow.log_metric('accuracy_score',acc)
    mlflow.log_metric('recall_score',recall)
    mlflow.log_metric('f1_score',f1)
    mlflow.log_metric('precision_score',precision)

    sns.heatmap(cm,annot=True,fmt='.2f')
    plt.savefig('cm.png')
    mlflow.log_artifact('cm.png')

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(rfc,'random_forest_classifier')

    mlflow.set_tag('author','Jay Kanakia')
    mlflow.set_tag('experiment_type','sample')