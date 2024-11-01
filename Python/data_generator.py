import numpy as np
import scipy as sp
import pandas as pd
import sklearn.model_selection
import datagen_modules

#Define the causal structure
def dataGen(seednum):
    # Fix the random seed
    np.random.seed(seednum)

    # Define variables
    Y = pd.read_csv('data_line4_stop2.csv',sep=',')
    # Y = Y.loc[Y.line_number == 4]
    X=pd.DataFrame()
    dictName = {}
    parentDict = {}
    dictName[0] = "Dwell_time"
    dictName[1] = "Scheduled_travel_time"
    dictName[2] = "Preceding_stop_delay"
    dictName[3] = "Origin_delay"

    dictName[4] = "Extra_cold"
    dictName[5] = "Weekend"
    dictName[6] = "AP"
    dictName[7] = "MP"
    dictName[8] = "Cold"

    dictName[9] = "Arrival_delay"


    X[0]=Y.Dwell_time
    X[1] = Y.Scheduled_travel_time
    X[2] = Y.Preceding_stop_delay
    X[3] = Y.Origin_delay

    X[4] = Y.Extra_cold
    X[5] = Y.weekend
    X[6] = Y.AP
    X[7] = Y.MP
    X[8] = Y.Cold

    X[9] = Y.Arrival_delay


    ########################################################################
    # Topo order
    ########################################################################
    def deriveTopoOrder(X):
        g = datagen_modules.Graph(len(X.columns))
        for x in range(len(X.columns)):
            parentDict[x] = []

        g.addEdge(5, 0); parentDict[0].append(5)
        g.addEdge(6, 0); parentDict[0].append(6)
        g.addEdge(7, 0); parentDict[0].append(7)
        g.addEdge(8, 0); parentDict[0].append(8)

        g.addEdge(6, 1); parentDict[1].append(6)
        g.addEdge(5, 1); parentDict[1].append(5)
        g.addEdge(7, 1); parentDict[1].append(7)

        g.addEdge(3, 2); parentDict[2].append(3)
        g.addEdge(6, 2); parentDict[2].append(6)
        g.addEdge(5, 2); parentDict[2].append(5)
        g.addEdge(8, 2); parentDict[2].append(8)

        g.addEdge(6, 3); parentDict[3].append(6)
        g.addEdge(4, 3); parentDict[3].append(4)
        g.addEdge(8, 3); parentDict[3].append(8)
        g.addEdge(5, 3); parentDict[3].append(5)

        g.addEdge(0, 9); parentDict[9].append(0)
        g.addEdge(1, 9); parentDict[9].append(1)
        g.addEdge(2, 9); parentDict[9].append(2)
        g.addEdge(3, 9); parentDict[9].append(3)
        g.addEdge(5, 9); parentDict[9].append(5)


        topoSort = g.topologicalSort()
        return topoSort
    topoSort = deriveTopoOrder(X)

    ########################################################################
    # ordered set
    ########################################################################
    ordered_columns = [X.columns.to_list()[x] for x in topoSort]
    X = X[ordered_columns]

    return X, topoSort, dictName, parentDict

if __name__ == '__main__':
    print("hello, world")
    # n = 100
    seednum = 123

    # X, topoSort, dictName, parentDict = dataGen(n,seednum)
    X, topoSort, dictName, parentDict = dataGen(seednum)
    print('X',X)
    print('topoSort', topoSort)
    print('dictName', dictName)
    print('parentDict', parentDict)
    X_train, y_train, X_test, y_test=datagen_modules.dataSplit(X,seednum)
    # X_train.to_csv('train1.csv')
    # X_test.to_csv('Test1.csv')
    # # print('X',X)


