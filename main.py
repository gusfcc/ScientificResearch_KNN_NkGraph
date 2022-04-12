import sys
import os
import numpy as np
import pandas as pd
import sklearn as sk
import scipy as sp
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold
from sklearn.metrics import confusion_matrix as CM
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rdm
import math as m
import networkx as nx
import random as rng
from NKGraph import NKGraph 

def main():
    Ki = 5
    Kf = 5
    p = 10
    
    createFiles = True
    
    pd.set_option('display.max_rows', 10)

    np.set_printoptions(threshold = sys.maxsize)
        
    
    resultsF = open("Results.txt", "w")
    for file in os.scandir("DataSets2"):        
        fFullPath = file.path
        fPath = os.path.splitext(fFullPath)[0]
        fName = os.path.split(os.path.splitext(fFullPath)[0])[1]
        dir = fName + "Files"        
        
        if(not os.path.isdir(dir) and createFiles is True):
            os.mkdir(dir)        
        
        nk = NKGraph(fName)
        
        if (createFiles is True):        
            nk.distFile(nk.distances, dir + '/' + fName + "_dist.txt")
        
        resultsF.write(fName + '\n')
        
        for k in range(Ki, Kf + 1):
            alpha = 2/k
            print(alpha)
        
            nkgraph = nk.createNKGraph(k, alpha, nk.distances, nk.dt, nk.density)
            
            if (createFiles is True):
                nk.graphAdjFile(nkgraph, fPath  = dir + '/' + fName + "_" + str(k) + "_NKG.txt")
        
            resultsF.write(str(k) + '\n')
            c1 = nk.nkCrossValKFold(k, alpha, p, type = 'a', createFiles = createFiles, fPath = dir)
            total1 = np.sum(c1)
            
            correct1 = 0
            for i in range(0,np.shape(c1)[1]):            
                correct1 += c1[i,i]
                
            resultsF.write(str(c1) + '\n')
            resultsF.write(str(correct1) + '/' + str(total1) + " = " + str(correct1/total1) + '\n')
            
            #---
            
            c2 = nk.nkCrossValKFold(k, alpha, p, type = 'b', createFiles = createFiles, fPath = dir)
            total2 = np.sum(c2) 
            
            correct2 = 0
            for i in range(0,np.shape(c2)[1]):            
                correct2 += c2[i,i]
                
            resultsF.write(str(c2) + '\n')
            resultsF.write(str(correct2) + '/' + str(total2) + " = " + str(correct2/total2) + '\n')
            
            #---
            
            c3 = nk.KNNCrossValKFold(k, p, createFiles = createFiles, fPath = dir)
            total3 = np.sum(c3)
            
            correct3 = 0
            for i in range(0,np.shape(c3)[1]):            
                correct3 += c3[i,i]
            
            resultsF.write(str(c3) + '\n')
            resultsF.write(str(correct3) + '/' + str(total3) + " = " + str(correct3/total3) + '\n')
            
        resultsF.write('\n\n')
        
    
    resultsF.close()
    
    '''
    dir = fName + "Files"
    
    if(not os.path.isdir(dir)):
        os.mkdir(dir)
    
    
    c1 = nkCrossValKFold(K, p, distances, dt, dtl, type = 'a', dir = dir)
    print(c1)
    print()
    c2 = nkCrossValKFold(K, p, distances, dt, dtl, type = 'b', dir = dir)
    print(c2)
    print()
    c3 = KNNCrossValKFold(K, p, distances, dt, dtl, dir = dir)
    print(c3)
    '''
        
if __name__ == '__main__':
    main()