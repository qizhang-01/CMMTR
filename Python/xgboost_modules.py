import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import xgboost
import random
import data_generator
import datagen_modules
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
np.random.seed(42)
random.seed(42)

def checkMode(vec):
    if len(pd.unique(vec)) == 2:
        return "binary"
    elif len(pd.unique(vec)) > 2 and len(pd.unique(vec)) < 10:
        return "multi"
    else:
        return "continuous"

def fit_xgboost(X, y, regtype):
    """ Train an XGBoost model with early stopping.
    regtype
    * 0: binary
    * 1: multi-class
    * 2: continuous
    """
    if regtype == "multi" or regtype == "binary":
        label = y.copy()
        y_unique = sorted(pd.unique(y))
        for idx in range(len(y_unique)):
            yvar = y_unique[idx]
            label[label == yvar] = int(idx)
    else:
        label = y.copy()

    # X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X, label)
    X_train, X_test, y_train, y_test = X,X,label,label
    # print('X_train',X_train)
    # print('y_train', y_train)
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    if regtype == 'binary':
        model = xgboost.train(
            { "eta": 0.5, "subsample": 0.5, "max_depth": 3, "objective": "reg:logistic","seed": 42}, dtrain, num_boost_round=20,
            evals=((dtest, "test"),), early_stopping_rounds=20, verbose_eval=False
        )
        return model
    elif regtype == 'multi':
        model = xgboost.train(
            {"eta": 0.5, "subsample": 0.5, "max_depth": 3, "objective": "multi:softprob",
             "num_class": len(pd.unique(label)), "eval_metric":'mlogloss',"seed": 42}, dtrain,
            evals=((dtest, "test"),), early_stopping_rounds=20, verbose_eval=False, num_boost_round=20,
            evals_result={'eval_metric': 'merror'}
        )
        return model
    elif regtype == 'continuous':
        model = xgboost.train(
            {"eta": 0.5, "subsample": 0.5, "max_depth": 3, "objective": "reg:squarederror","seed": 42}, dtrain, num_boost_round=20,
            evals=((dtest, "test"),), early_stopping_rounds=20, verbose_eval=False
        )
        return model

def predict_xgboost(model, X, y, regtype):

    if regtype == "multi" or regtype == "binary":
        label = y.copy()
        y_unique = sorted(pd.unique(y))
        for idx in range(len(y_unique)):
            yvar = y_unique[idx]
            label[label == yvar] = idx
    else:
        # inverse label
        label = y.copy()

    if regtype == 'binary':
        pred = model.predict(xgboost.DMatrix(X))
        pred = label * pred + (1 - label) * (1 - pred)
        return pred
    if regtype == 'multi':
        pred = model.predict(xgboost.DMatrix(X))
        label = pd.DataFrame(label).reset_index(drop=True)
        pred = np.asarray([pred[idx][int(label.iloc[idx])] for idx in range(X.shape[0])])
        return pred
    if regtype == 'continuous':
        pred = model.predict(xgboost.DMatrix(X))
        rmse = np.sqrt(mean_squared_error(y, pred))
        return pred
if __name__ == '__main__':
    print("hello, world")
    # n = 100
    seednum = 42

    # X, topoSort, dictName, parentDict = dataGen(n,seednum)
    X, topoSort, dictName, parentDict = data_generator_1.dataGen(seednum)
    # print('X',X)
    print('topoSort', topoSort)
    print('dictName', dictName)
    print('parentDict', parentDict)
    X_train, y_train, X_test, y_test=datagen_modules.dataSplit(X,seednum)
    model= fit_xgboost(X_train,y_train,'continuous')
    # prediction, rmse=predict_xgboost(model,X_test,y_test,'continuous')
    # print(rmse)




    # X_train.to_csv('train1.csv')
    # X_test.to_csv('Test1.csv')
    # print('X',X)

