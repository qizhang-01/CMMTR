import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

# Python program to print topological sorting of a DAG
from collections import defaultdict

# Class to represent a graph
class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # No. of vertices

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Print contents of stack
        return(stack)

def normalizer(dictvec):
    dictvec_copy = dictvec.copy()  # Make a copy of the dictionary
    normalized_var = np.sum(list(dictvec_copy.values()))
    for key in dictvec_copy.keys():
        dictvec_copy[key] /= normalized_var
    # dictvec /= np.sum(list(dictvec.values()))
    return dictvec_copy

def discretizer(pdvec,numbins):
    # label = np.array(list(range(0,numbins)))
    # bins = np.linspace(np.min(pdvec) - 1e-5, np.max(pdvec)+1e-5, numbins+1, endpoint=True)
    # pdvec_discount = pd.to_numeric(pd.cut(pdvec, bins = bins, labels = label))

    bins = np.linspace(np.min(pdvec) - 1e-5, np.max(pdvec) + 1e-5, numbins + 1, endpoint=True)
    label = [(bins[idx] + bins[idx-1])/2 for idx in range(1,len(bins))]
    pdvec_discrete = pd.to_numeric(pd.cut(pdvec, bins = bins, labels = label))
    return pdvec_discrete

def sort_on_reference(ref,vec):
    returnVec = []
    for val in ref:
        if val in vec:
            returnVec.append(val)
        else:
            continue
    return returnVec

def dataSplit(X,seednum):
    np.random.seed(seednum)
    print("Spliting on...")
    groups = X.groupby([8])

    train_frames = []
    test_frames = []

    for name, group in groups:
        # Split each group into train and test
        if len(group) > 1:
            train_group, test_group = train_test_split(group, test_size=0.5, random_state=42)
            train_frames.append(train_group)
            test_frames.append(test_group)

    # Concatenate all train and test groups back together
    Train = pd.concat(train_frames)
    Test = pd.concat(test_frames)



    # # Train, Test = sklearn.model_selection.train_test_split(X, test_size=0.5)
    # num_rows = int(len(X) * 0.5)
    # Train=X.iloc[:num_rows]
    # Test=X.iloc[num_rows:]

    # while 1:
    #     Train, Test = sklearn.model_selection.train_test_split(X,test_size=0.5)
    #     stopSwitch = True
    #     for colidx in X.columns[:-1]:
    #         if sorted( pd.unique( Train[colidx] ) ) != sorted( pd.unique( Test[colidx] ) ):
    #             stopSwitch = False
    #             break
    #     if stopSwitch:
    #         break
    #     else:
    #         continue
    print("Spliting done!")

    X_train = Train[X.columns[:-1]].copy()
    y_train = Train[X.columns[-1]].copy()

    X_test = Test[X.columns[:-1]].copy()
    y_test = Test[X.columns[-1]].copy()

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':

    print("datagen_modules.py")