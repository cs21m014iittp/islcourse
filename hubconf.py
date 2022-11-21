import torch
import numpy as np
import sklearn
from sklearn.datasets import make_blobs,make_circles
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



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
