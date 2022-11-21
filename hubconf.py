import torch
import numpy as np
import sklearn
from sklearn.datasets import make_blobs,make_circles,load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score,completeness_score,v_measure_score



def get_data_blobs(n_points=100):
  # write your code here
  # Refer to sklearn data sets
  X, y = make_blobs(n_samples=n_points)
  # write your code ...
  return X,y

def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_circles(n_samples=n_points)
  # write your code ...
  return X,y

def get_data_mnist():
  # write your code here
  # Refer to sklearn data sets
  data = load_digits()
  X = data.images
  y = data.target
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  km = KMeans(n_clusters=k) # this is the KMeans object
  # write your code ...
  km.fit(X)
  return km


def assign_kmeans(km=None,X=None):
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = 0,0,0 # you need to write your code to find proper values

  h = homogeneity_score(ypred_1,ypred_2)
  c = completeness_score(ypred_1,ypred_2)
  v = v_measure_score(ypred_1,ypred_2)
  return h,c,v


def build_lr_model(X=None, y=None):
  lr_model = LogisticRegression()
  # write your code...
  # Build logistic regression, refer to sklearn
  if X.ndim > 2:
      n_samples = len(X)
      X= X.reshape((n_samples, -1))
  lr_model.fit(X,y)
  return lr_model

def build_rf_model(X=None, y=None):
  rf_model = RandomForestClassifier()
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  if X.ndim > 2:
      n_samples = len(X)
      X= X.reshape((n_samples, -1))
  rf_model.fit(X,y)
  return rf_model


def get_metrics(model=None,X=None,y=None):
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  
  if X.ndim > 2:
      n_samples = len(X)
      X= X.reshape((n_samples, -1))
  classes = set()
  for i in y:
      classes.add(i)
  num_classes = len(classes)

  ypred = model.predict(X)
  acc, prec, rec, f1, auc = 0,0,0,0,0
  # write your code here...
  acc = accuracy_score(y,ypred)
  if num_classes == 2:
    prec = precision_score(y,ypred)
    recall = recall_score(y,ypred)
    f1 = f1_score(y,ypred)
    auc = roc_auc_score(y,ypred)

  else:
    prec = precision_score(y,ypred,average='macro')
    recall = recall_score(y,ypred,average='macro')
    f1 = f1_score(y,ypred,average='macro')
    pred_prob = model.predict_proba(X)
    roc_auc_score(y, pred_prob, multi_class='ovr')
    #auc = roc_auc_score(y,ypred,average='macro',multi_class='ovr')

  return acc, prec, rec, f1, auc
