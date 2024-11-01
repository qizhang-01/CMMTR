import numpy as np
import pandas as pd
import scipy as sp
import xgboost

import datagen_modules
import xgboost_modules
import data_generator
import semopy
from sklearn.linear_model import LinearRegression

# generate noise
def generate_noise(addedVar, noise_mean, noise_sigma, noiseTF):
    # NOISE
    if noiseTF: 
        noise = np.random.normal(loc=noise_mean, scale=noise_sigma, size=addedVar.shape)
    else:
        noise = 0
    return noise

# Train models for E[Y | v] using trainDataset
def train_Y(X_train,y_train):
    # Compute theta_{k,2} = E[\theta_k1 | Vk, pre(Vk)]
    regmode = xgboost_modules.checkMode(y_train)
    model = xgboost_modules.fit_xgboost(X=X_train, y=y_train, regtype=regmode)
    return model



#using linear regression t
# def train_Y(X_train, y_train):
#     # Initialize the Linear Regression model
#     model = LinearRegression()
#
#     # Fit the model to the training data
#     model.fit(X_train, y_train)
#
#     return model
#
# def predict_Y(model, X,y):
#     pred=model.predict(X)
#     return pred

# def train_Y(X_train, y_train):
#     # np.random.seed(seednum)
#     # Model specification: This is a crucial step and will depend on your study.
#     # The example below is just a placeholder. You need to define your own SEM model here.
#     column_names = ['var5', 'var1', 'var4', 'var3', 'var2', 'var0']  # Example names
#     X_train.columns = column_names
#     y_train.name = 'Target'
#
#     model_desc = """
#     # Define latent variables and their indicators (if any)
#     var0 ~ var5
#     var2 ~ var5+var3
#     var3 ~ var5+var4
#     var4 ~ var1
#     # Define regressions
#     Target ~ var0+var2+var3+var4+var1
#     """
#
#     # Convert X_train to a pandas DataFrame if it's not already, and make sure y_train is included in the DataFrame
#     # Assuming X_train is a DataFrame and y_train is a series aligned with X_train's index
#     data = pd.concat([X_train, y_train], axis=1)
#     # print(data)
#     # data = X_train.copy()
#     # data['y_train'] = y_train  # Make sure 'y_train' matches the variable name in model_desc
#
#     # Instantiate and fit the SEM model
#     model = semopy.Model(model_desc)
#     model = model.fit(data)
#
#     return model




# Train models for {P(Vi | pre(Vi))} using trainDataset
def train_condProb(X_train, topoSort,parentDict):
    '''
        Output: allCondProb for trained dataset
        Train with P(vi | pre(vi))
    '''
    pred_models = {}
    for idx in range(0,5):

        # For variables in topoSort
        Vi = X_train[topoSort[idx]]

        # Take a unique values
        uniqueVi = sorted(pd.unique(Vi))


        # Compute P(vi) for all vi
        prob = np.array([np.sum(Vi == x) / len(Vi) for x in uniqueVi])

        # append pred_models
        pred_models[topoSort[idx]] = prob

    for idx in range(5,len(topoSort)-1):
        # Set Vi
        varidx = topoSort[idx]
        Vi = X_train[varidx]
        # print('vi',Vi)

        # pre(Vi)
        pre_Vi = X_train[parentDict[varidx]]
        # print('pre_Vi',pre_Vi)

        regmode = xgboost_modules.checkMode(Vi)
        model = xgboost_modules.fit_xgboost(X=pre_Vi, y=Vi, regtype=regmode)

        #linear regression
        # model= train_Y(pre_Vi, Vi)

        pred_models[varidx] = model

    return pred_models


# Evaluate for {P(Vi | pre(Vi))} for all samples in the Test
def estimate_condProb(X_test, allCondModel, topoSort, parentDict):
    '''
    Output: allCondProb
    '''

    eval_models = {}

    ''' For V0 and V1
    '''
    for idx in range(0,5):
        # Call the model
        prob = allCondModel[topoSort[idx]]
        # print('prob',prob)

        # For variables in topoSort
        Vi = X_test[topoSort[idx]]
        # print(Vi)

        # Take a unique values
        uniqueVi = sorted(pd.unique(Vi))
        # print('len(uniqueVi)',len(uniqueVi))

        # Compute P(vi) for all vi
        prob_Vi = np.asarray([prob[uniqueVi.index(x)] for x in Vi])
        eval_models[topoSort[idx]] = prob_Vi

    for idx in range(5, len(topoSort)-1):
        # Set Vi
        varidx = topoSort[idx]
        Vi = X_test[varidx]

        # Call the condmodel P(Vi | pre(Vi))
        model = allCondModel[varidx]

        # Set pre(Vi) from test dataset
        pre_Vi = X_test[parentDict[varidx]]

        # Evalute P(Vi | pre(Vi))
        regmode = xgboost_modules.checkMode(Vi)
        prob_Vi = xgboost_modules.predict_xgboost(model, X=pre_Vi, y=(Vi), regtype=regmode)

        #Linear regression
        # prob_Vi = predict_Y(model, X=pre_Vi, y=(Vi))

        # append pred_models
        eval_models[varidx] = prob_Vi

    # Add noises
    #修改
    # for k in eval_models.keys():
    #     eval_models[k] += generate_noise(eval_models[k], X_test.shape[0] ** (-1 / 4), 0.1,noiseTF)

    return eval_models

# It encodes the subset S as a key string.
# All subset S takes the same key if they contain the same element.
def keygen(S_sorted):
    return ''.join([str(x) for x in S_sorted])



# Given S, compute \omega_{S} := \prod_{i \in S} I_{vi}(Vi) / P(Vi | pa(Vi)).
def constructOmega_S(X_test, x, allCondProb, S, topoSort):
    '''
    Input
        X_test: A data for the evaluation
        x: This is v_{S}
        allCondProb: P(Vi | pa(Vi)) evaluated from X_test
        S: A set S
        topoSort: Topological order

    Output:
        omega: A dictionary key:[S1,S2,...], output \omega_{Si} := \prod_{i \in Si} I_{vi}(Vi) / P(Vi | pa(Vi)) for each Si
    '''

    eps = 1e-4
    # eps = 0
    omega = {}
    # Sorting S w.r.t topo order
    S_sorted = datagen_modules.sort_on_reference(ref=topoSort, vec=S)

    # 1 vector
    prev_omega = np.repeat(1,X_test.shape[0])

    # For each v_{Si} \in v_{S}
    for idx in range(len(S_sorted)):
        # k = Si
        k = S_sorted[idx]

        ''' For each V_{S_i} \in V_{S}
        '''
        # Check Ck := { V_{Sk}+1, ..., V_{S_{k+1}} - 1} Empty
        if idx == len(S_sorted)-1:
            Ck = topoSort[topoSort.index(k) + 1:]
        else:
            k_plus_1 = S_sorted[idx + 1]
            Ck = topoSort[topoSort.index(k) + 1:topoSort.index(k_plus_1)]
        # If Ck = empty, then do nothing
        if Ck == []:
            continue

        # Identify P(Vk | pre(Vk))
        pred_k = allCondProb[k]

        # Compute I_{vk}(Vk)
        vk = x[k].values[0]
        Vk = X_test[k]
        Ivk_Vk = ((Vk == vk)*1)

        # Compute \omega_{k} = \omega_{k-1} * ( I_{vk}(Vk)/P(Vk | pa(Vk)) )
        newVal = (Ivk_Vk/(pred_k + eps))
        clip_var = 3
        # newVal[newVal > clip_var ] = clip_var
        one_sigma_newVal_lower = np.mean(newVal) - np.std(newVal)
        one_sigma_newVal_upper = np.mean(newVal) + np.std(newVal)
        newVal[newVal > one_sigma_newVal_upper] = one_sigma_newVal_upper  + np.random.normal(loc=0,scale=0.1,size=len(newVal))
        newVal[newVal < one_sigma_newVal_lower] = one_sigma_newVal_lower + np.random.normal(loc=0, scale=0.1, size=len(newVal))

        newVal_prev = newVal * prev_omega
        one_sigma_newVal_lower = np.mean(newVal_prev ) - np.std(newVal_prev )
        one_sigma_newVal_upper = np.mean(newVal_prev ) + np.std(newVal_prev )
        newVal_prev[newVal_prev > one_sigma_newVal_upper] = one_sigma_newVal_upper + np.random.normal(loc=0, scale=0.1,size=len(newVal))
        newVal_prev[newVal_prev < one_sigma_newVal_lower] = one_sigma_newVal_lower + np.random.normal(loc=0, scale=0.1,size=len(newVal))
        # newVal_prev[newVal_prev > clip_var] = clip_var

        omega[k] = newVal_prev

        prev_omega = omega[k]

    ''' omega[k] = \prod_{i \in S; and i \prec k}
    '''

    return omega

# Compute \theta_{k,1}, \theta_{k,2} for E[Y | do(v_S)]
def constructTheta_S(X_train, y_train, X_test, y_test, x, S, topoSort):
    '''
    Input
        X_train:
        y_train:
        X_test:
        y_test:
        x:
        S:

    Output
        \theta_{k,1}, \theta_{k,2} for k \in S
    '''

    N = X_test.shape[0]

    # Sort S based on the topological error.
    S_sorted = datagen_modules.sort_on_reference(ref=topoSort, vec=S)
    S_sorted_reverse = S_sorted.copy()
    S_sorted_reverse.reverse()

    theta_1 = dict()
    theta_2 = dict()

    theta_k1_train = y_train
    theta_k1_test = y_test

    # for k in S_sorted_reverse: S_{m}, S_{m-1},...
    for idx in range(len(S_sorted_reverse)):
        k = S_sorted_reverse[idx]

        # Check Ck := { V_{Sk}+1, ..., V_{S_{k+1}} - 1} Empty
        if idx == 0:
            Ck = topoSort[topoSort.index(k) + 1:]
        else:
            k_plus_1 = S_sorted_reverse[idx-1]
            Ck = topoSort[topoSort.index(k)+1:topoSort.index(k_plus_1)]

        if Ck == []:
            continue

        # Ck = topoSort[topoSort.index(k)+1:]
        theta_1[k] = theta_k1_test

        # Vk and Predecessor of Vk; Vk will be regressed onto {Predecessor of Vk}.
        colchoice = [val for val in topoSort if topoSort.index(val) <= topoSort.index(k)]

        X_choice_train = X_train[colchoice]
        X_choice_test = X_test[colchoice]

        # Compute theta_{k,2} = E[\theta_k1 | Vk, pre(Vk)]
        regmode = xgboost_modules.checkMode(theta_k1_train)
        model = xgboost_modules.fit_xgboost(X=X_choice_train, y=theta_k1_train, regtype=regmode)

        #Linear regression
        # model=LinearRegression().fit(X=X_choice_train, y=theta_k1_train)

        if regmode == "binary":
            theta_k2_test = model.predict(xgboost.DMatrix(X_choice_test))
            # theta_k2_train = model.predict(xgboost.DMatrix(X_choice_train))
        else:
            theta_k2_test = xgboost_modules.predict_xgboost(model,X_choice_test,theta_k1_test,regmode)

            #Linear regression
            # theta_k2_test = predict_Y(model, X_choice_test, theta_k1_test)

        theta_2[k] = theta_k2_test

        # Compute theta_{k-1,1} = E[\theta_k1 | vk, pre(Vk)]
        X_choice_train = X_choice_train.copy()
        X_choice_test = X_choice_test.copy()
        X_choice_train[k] = np.repeat(x[k].values[0],X_choice_train.shape[0])
        X_choice_test[k] = np.repeat(x[k].values[0], X_choice_test.shape[0])

        if regmode == "binary":
            theta_k1_test = model.predict(xgboost.DMatrix(X_choice_test))
            theta_k1_train = model.predict(xgboost.DMatrix(X_choice_train))
        else:
            theta_k1_test = xgboost_modules.predict_xgboost(model,X_choice_test,theta_k1_test,regmode)
            theta_k1_train = xgboost_modules.predict_xgboost(model,X_choice_train,theta_k1_train,regmode)

            #Linear regression
            # theta_k1_test = predict_Y(model, X_choice_test, theta_k1_test)
            # theta_k1_train =predict_Y(model, X_choice_train, theta_k1_train)

    # This is \theta_{0,1}
    theta_1[-1] = theta_k1_test

    # Noise
    #修改
    theta_1[-1] += generate_noise(theta_1[-1],np.mean(y_train)*(N**(-1/4))*10,1, 0)

    # for k in theta_1.keys():
    #     theta_1[k] += N ** (-1 / 3)

    return theta_1, theta_2

################################################################################################
################################################################################################
# Double/Debiased Machine Learning
################################################################################################
################################################################################################

################################################################################################
# Case 1
################################################################################################

# Given \omega_k and \theta_{k,1}, \theta_{k,2} for all k \in S, compute the DML estimate
def computeDML_given_omega_theta(omegaDict,theta_1,theta_2):
    '''
    Input:
        omegaDict: From constructOmega_S. This is \bm{\omega}^S
        theta_1: From constructTheta_S. This is \bm{\theta^S_{1}}
        theta_2: From constructTheta_S. This is \bm{\theta^S_{2}}

    Output:
        DML estimates for E[Y | do(v_S)]
    '''
    # theta^S_{0,1}
    EIF = np.asarray(theta_1[-1].copy())

    # Sum to compute \omega_k * (\theta_{k,1} - \theta_{k,2})
    for k in omegaDict.keys():
        EIF += np.asarray(omegaDict[k]) * (np.asarray(theta_1[k]) - np.asarray(theta_2[k]))
    return np.mean(EIF)

# Compute the DML estimator for v_{S}; i.e., Compute E[Y | do(V_{S} = v_{S})] using DML
def computeDML_S(X_list,y_list, allCondProb_list, x, S, topoSort, DMLDict):
    '''
    Input
        X_list: [X_{train}, X_{test}]
        y_list: [y_{train}, y_{test}]
        allCondProb_list: [allCondProb_eval_train, allCondProb_eval_test]
        x: Fixed v_{S}
        S: a subset S
        topoSort:
        DMLDict: key: S, value: DML estimates

    Output
        DMLDict
    '''

    # If S = 0 or S = [n], do nothing
    if len(S) == 0 or len(S) == len(X_list[0].columns):
        DMLDict[keygen(S)] = 1
        return DMLDict

    # Sort S w.r.t. topoOrder
    S_sorted = datagen_modules.sort_on_reference(ref=topoSort, vec=S)

    # If S is in DMLDict, then just return DMLDict
    if keygen(S_sorted) in DMLDict.keys():
        return DMLDict

    result = 0
    for k in [0,1]:
        # Compute \bm{\omega^S};
        ## Note allCondProb is trained with X_list[1-k] and evaluated with X_list[k]
        omegaDict = constructOmega_S(X_test =X_list[k], x=x, allCondProb=allCondProb_list[k], S=S, topoSort=topoSort)

        # Compute \bm{\theta}^S_{0}, \bm{\theta}^S_{1}
        theta_1, theta_2 = constructTheta_S(X_train = X_list[1-k], y_train = y_list[1-k], X_test=X_list[k], y_test = y_list[k], x=x, S=S, topoSort=topoSort)
        result += computeDML_given_omega_theta(omegaDict, theta_1, theta_2)

    result /= 2
    DMLDict[keygen(S_sorted)] = result
    return DMLDict


if __name__ == '__main__':
    n = 1000
    seednum = 123
    targetidx = 100

    X, topoSort, dictName, parentDict = data_generator_1.dataGen(seednum)
    X_train, y_train, X_test, y_test = datagen_modules.dataSplit(X, seednum)
    x = X_test.iloc[[targetidx]]
    y = y_test.iloc[[targetidx]].values[0]
    print('x',x)
    print('y', y)

    X_list = [X_train, X_test]
    y_list = [y_train, y_test]

    S = [2]
    model_Y_train = train_Y(X_train,y_train)
    print(model_Y_train)
    # model_Y_test = train_Y(X_test,y_test)
    # model_Y_list = [model_Y_train, model_Y_test]
    # DMLDict = {}
    # noiseTF=None
    # DMLDict = computeDML_S(X_list, y_list, model_Y_list, x, S, topoSort, DMLDict)
    # print(DMLDict)

    # IPWDict = {}
    # IPWDIct = computeIPW_S(X_list, y_list, x, S, topoSort, IPWDict,noiseTF)
    # print(IPWDIct)
    #
    # PIDict = {}
    # PIDict = computePI_S(X_list, y_list, model_Y_list, x, S, topoSort, IPWDict,noiseTF)
    # print(PIDict)


