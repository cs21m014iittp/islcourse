import torch
import numpy as np
import sklearn
from sklearn.datasets import make_blobs,make_circles,load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score,completeness_score,v_measure_score,accuracy_score, precision_score,recall_score,f1_score,roc_auc_score
from torch import nn
from torchvision.transforms import ToTensor
import torch.nn.functional as Fun

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = {'penalty' : ['l1','l2']}
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = { 'n_estimators' : [1,10,100],'criterion' :["gini", "entropy"], 'max_depth' : [1,10,None]  }
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose

  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
  top1_scores = []
  
  if X.ndim > 2:
      n_samples = len(X)
      X= X.reshape((n_samples, -1))
      
  for score in metrics:
      grid_search_cv = GridSearchCV(model,param_grid,scoring = score,cv=cv)
      grid_search_cv.fit(X,y)
      top1_scores.append(grid_search_cv.best_estimator_.get_params())
      
  return top1_scores


class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self).__init__()
    self.num_classes = num_classes
    self.fc_encoder = nn.Linear(inp_dim,hid_dim).to(device) # write your code inp_dim to hid_dim mapper
    self.fc_decoder = nn.Linear(hid_dim,inp_dim).to(device) # write your code hid_dim to inp_dim mapper
    self.fc_classifier = nn.Linear(hid_dim,num_classes).to(device) # write your code to map hid_dim to num_classes
    
    self.relu = nn.ReLU() #write your code - relu object
    self.softmax = nn.Softmax() #write your code - softmax object
    
  def forward(self,x):
    if x.ndim > 2:
        flat = nn.Flatten()
        x = flat(x) # write your code - flatten x
    else:
        flat = nn.Flatten(start_dim=0)
        x = flat(x)

    print(x.shape)
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding los
  def loss_fn(self,x,yground,y_pred,xencdec):
      # class prediction loss
      # yground needs to be one hot encoded - write your code
      # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
      #print(yground)

      tmp = Fun.one_hot(yground, num_classes= self.num_classes).to(device)
      y_pred , tmp = y_pred.to(device) , tmp.to(device)
      v = -(tmp * torch.log(y_pred + 0.0001))
      lc1 = torch.mean(v)
      
      # auto encoding loss
      if x.ndim > 2:
          flat = nn.Flatten()
          x = flat(x) # write your code - flatten x
      else:
          flat = nn.Flatten(start_dim=0)
          x = flat(x)
      
      lc2 = torch.mean((x - xencdec)**2)
      lval = lc1 + lc2
      return lval
 
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  X_np, y_np = get_data_mnist()

  X = torch.tensor(X_np)
  y = torch.tensor(y_np)
  # write your code
  return X,y

def get_loss_on_single_point(mynn,x0,y0):
    y_pred, xencdec = mynn(x0)
    lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
    # the lossval should have grad_fn attribute set
    return lossval

def train_combined_encdec_predictor(mynn,X,y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = torch.optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
      optimizer.zero_grad()
      ypred, Xencdec = mynn(X)
      lval = mynn.loss_fn(X,y,ypred,Xencdec)
      lval.backward()
      optimzer.step()
    
  return mynn
