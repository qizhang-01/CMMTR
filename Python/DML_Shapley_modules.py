import numpy as np
import multiprocess as mp

import datagen_modules
import DML_modules
import data_generator
import pandas as pd
import matplotlib.pyplot as plt
import sys 

# Generate random permutation of features of

def generateRandomPerm(X):
    # Making a random seed
    np.random.seed()
    nlist = np.sort(np.asarray(X.columns))
    nlist_random = np.random.permutation(nlist)
    Slist = []
    for idx in range(len(nlist_random)+1):
        Slist.append( nlist_random[:idx] )
    return Slist

def stopCriterion(accumDict, curDict, stop_value):
    for key in accumDict.keys():
        if np.abs( accumDict[key] - curDict[key] ) > stop_value:
            return False
    return True

# Combine individual permutation results
def combineResult(results):
    result_dict = dict()
    for key in results[0].keys():
        result_dict[key] = 0

    for idx in range(len(results)):
        for key in results[idx].keys():
            result_dict[key] = ((idx * result_dict[key]) + results[idx][key]) / (idx+1)
    return result_dict

##########################################################################################
##########################################################################################
# DML
##########################################################################################
##########################################################################################

##########################################################################################
# Case 1
##########################################################################################
# Update of DMLDict(S) for individual permutation
def Shapley_update_perm(X_list, y_list, allCondProb_list, x, y, Slist, topoSort, DMLDict):
    '''
    Output:
        E[Y | do(v_{S1})] - E[Y | do(v_{S0})]
        Updated DMLDict
    '''
    # Dictionary Dict(Var) for individual permutation
    shapley_dict_perm = dict()
    # For each S generated for the individual permutation
    for idx in range(1,len(X_list[0].columns)+1):
        # pre_{\pi}(Vi)
        S0 = Slist[idx-1]
        # Vi and pre_{\pi}(Vi)
        S1 = Slist[idx]
        # Vi
        updated_variable = np.setdiff1d(S1,S0)[0]

        # If S0 = emptyset,
        if len(S0) == 0:
            # E[Y]
            result_S0 = (np.mean(y_list[0]) + np.mean(y_list[1]))/2
        else:
            # Update the DMLDict for S0
            DMLDict = DML_modules.computeDML_S(X_list, y_list, allCondProb_list, x, S0, topoSort, DMLDict)
            # Have E[Y | do(v_{S0})]
            S0_sorted = datagen_modules.sort_on_reference(ref=topoSort, vec=S0)
            result_S0 = DMLDict[DML_modules.keygen(S0_sorted)]

        # If S1 = [n], give y
        if len(S1) == len(X_list[0].columns):
            result_S1 = y
        else:
            # Update the DMLDict for S1
            DMLDict = DML_modules.computeDML_S(X_list, y_list, allCondProb_list, x, S1, topoSort, DMLDict)
            # Have E[Y | do(v_{S1})]
            S1_sorted = datagen_modules.sort_on_reference(ref=topoSort, vec=S1)
            result_S1 = DMLDict[DML_modules.keygen(S1_sorted)]

        # Compute E[Y | do(v_{S1})] - E[Y | do(v_{S0})]
        shapley_dict_perm[updated_variable] = result_S1 - result_S0

    return shapley_dict_perm, DMLDict

def computeShapley_DML(X_train, y_train, X_test, y_test, x, y, parentDict, topoSort, numiter):
    '''
    Output
        shapley_dict: Final Shapley Value
    '''
    # Initialize the DML_Dict
    DML_dict = dict()
    # Final Shapley Value
    shapley_dict = dict()

    # allCondModel trained using Train and Test dataset
    allCondModel_train = DML_modules.train_condProb(X_train, topoSort, parentDict)
    allCondModel_test = DML_modules.train_condProb(X_test, topoSort, parentDict)

    # Evaluated {P(Vi | pre(Vi)) } trained using Train and Test dataset
    allCondProb_eval_test = DML_modules.estimate_condProb(X_test, allCondModel_train, topoSort,parentDict)
    allCondProb_eval_train = DML_modules.estimate_condProb(X_train, allCondModel_test, topoSort, parentDict)
    X_list = [X_train, X_test]
    y_list = [y_train, y_test]
    allCondProb_list = [allCondProb_eval_train, allCondProb_eval_test]

    for key in X_train.columns:
        shapley_dict[key] = 0

    # Repeat the following:
    for idx in range(numiter):
        # Generate S
        Slist = generateRandomPerm(X_train)
        # print('Slist',Slist)
        # Generate an individual Shapley S and DML Dict.
        shapley_dict_S, DML_dict = Shapley_update_perm(X_list, y_list, allCondProb_list, x, y, Slist, topoSort, DML_dict)

        # Moving Average Update for the Shapley.
        for key in X_train.columns:
            shapley_dict[key] = ((idx * shapley_dict[key]) + shapley_dict_S[key]) / (idx+1)

    return shapley_dict

def multi_run_wrapper_DML(args):
    return computeShapley_DML(*args)

def Run_DML(seednum,numiter_per_sim,num_sim,targetidx,X, topoSort, parentDict,X_train, y_train, X_test, y_test):
    '''
    num_sim = number of parallelized worker
    numiter_per_sim = For each parallelized worker, the number of random permutation
    That is, the total number is num_sim *numiter_per_sim.
    '''
    # X, topoSort, dictName, parentDict = data_generator_1.dataGen(seednum)
    # X_train, y_train, X_test, y_test = datagen_modules.dataSplit(X, seednum)
    x1 = X.iloc[[targetidx]]
    y = x1[x1.columns[-1]].values[0]
    x = x1[x1.columns[:-1]]
    print('Test data_x:', x)
    print('Test data_y:', y)


    numiter = numiter_per_sim * num_sim

    resultShapley = computeShapley_DML(X_train, y_train, X_test, y_test, x, y, parentDict, topoSort, numiter)

    # myparam = (X_train, y_train, X_test, y_test, x, y, parentDict, topoSort, numiter_per_sim)
    #
    # pool = mp.Pool(processes=num_process)
    # results = pool.map(multi_run_wrapper_DML, [myparam] * num_sim)
    # resultShapley = combineResult(results)
    # for idx in range(len(X_train.columns)):
    #     print(dictName[idx], ":", np.round(resultShapley[idx], 4))
    return resultShapley




def plot_boxplot(data):
    # Mapping from index to real names
    dictName = {
        0: "Dwell_time",
        1: "Scheduled_travel_time",
        2: "Preceding_stop_delay",
        3: "Origin_delay",
        4: "Extra_cold",
        5: "Weekend",
        6: "AP",
        7: "MP",
        8: "Cold",
    }

    # Reorganize data into a format suitable for boxplot
    reorganized_data = {dictName[key]: [] for key in dictName}
    for entry in data:
        for key, value in entry.items():
            reorganized_data[dictName[key]].append(value)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.boxplot(reorganized_data.values(), labels=reorganized_data.keys())
    plt.xticks(rotation=45)
    plt.title('Boxplot of Various Delays')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()

if __name__ == '__main__':
    # n=10

    seednum = 42
    # targetidx = 20
    num_process = 5
    
    numiter_per_sim = 30
    num_sim = 1

    # casenum = 1
    noiseTF = 0


    X, topoSort, dictName, parentDict = data_generator_1.dataGen(seednum)
    X_train, y_train, X_test, y_test = datagen_modules.dataSplit(X, seednum)
    # x = X_test.iloc[[targetidx]]
    # y = y_test.iloc[[targetidx]].values[0]
    #
    # X_list = [X_train, X_test]
    # y_list = [y_train, y_test]
    # print(x,y)

    # resultDML = Run_DML(seednum, numiter_per_sim, num_sim, targetidx)
    # resultDML = datagen_modules.normalizer(resultDML)
    # print("DML", resultDML)
    targetidx = np.random.randint(0, 20000, size=100)
#     X.iloc[targetidx].to_csv('select_data_stop2_30_literate.csv')
    dml=[]
    dml_nonnor=[]
    for idx in targetidx:
        resultDML = Run_DML(seednum, numiter_per_sim, num_sim, idx,X, topoSort, parentDict,X_train, y_train, X_test, y_test)
        dml_nonnor.append(resultDML)
        print('resultDML',resultDML)
        resultDML1 = datagen_modules.normalizer(resultDML)
        print('resultDML1',resultDML1)
        dml.append(resultDML1)

    pd.DataFrame(dml).to_csv('resluts_data_stop2_nor_30_literate.csv')
    pd.DataFrame(dml_nonnor).to_csv('resluts_data_stop2_non_30_literate.csv')
    print(dml)
    plot_boxplot(dml)
