import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn 

from sklearn.ensemble import RandomForestclassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='onkar-git', repo_name='mlflow-dagshub-demo', mlflow=True)


mlflow.set_tracking_uri('https://dagshub.com/onkar-git/mlflow-dagshub-demo.mlflow')

iris = load_iris()

X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 10
n_estimators = 100

mlflow.set_experiment("iris-dt")

with mlflow.start_run():

    rf = RandomForestclassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_param("max_depth", max_depth)
    
    mlflow.log_metric("Accuracy", accuracy)

    cm = confusion_matrix(y_test,y_pred)

    sns.heatmap(cm,annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')


    plt.savefig('confusion_matrix.png')
    mlflow.log_artifacts('confusion_matrix.png')

    mlflow.log_artifacts(__file__)
    mlflow.sklearn.log_model(rf,"Random_Forest_model")


    mlflow.set_tag('author','nitish')
    mlflow.set_tag('project','iris-classification')
    mlflow.set_tag('algorithm','Random_Forest_model')

    
    

