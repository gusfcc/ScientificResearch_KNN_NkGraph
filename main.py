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
    Ki = 1  #Start K parameter [01]
    Kf = 10 #Final K parameter [10]
    Ai = 0  #Start alpha parameter [00]
    Af = 9  #Final alpha parameter [09]
    p = 10  #crosval p-fold [10]
    DSFolder = "Datasets" #Datasets folder [Datasets]
    createFiles = False #If all the steps on the creation of the graphs will be printed out [False]
    
    
    pd.set_option('display.max_rows', 10)

    np.set_printoptions(threshold = sys.maxsize)
        
    
    resultsF = open("Results.txt", "w").close()
    for file in os.scandir(DSFolder):        
        resultsF = open("Results.txt", "a")
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
            for a in range(Ai, Af +1):
                alpha = a/k
            
                nkgraph = nk.createNKGraph(k, alpha, nk.distances, nk.dt, nk.density)
                #nk.drawNKGraph(nkgraph, nk.dt)
                
                if (createFiles is True):
                    nk.graphAdjFile(nkgraph, fPath  = dir + '/' + fName + "_" + str(k) + '_' + str(a) + "_NKG.txt")
            
                resultsF.write('\n' + str(k) + '\n')
                resultsF.write(str(alpha) + ' - ' + str(a) + '\n')
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
 
        
if __name__ == '__main__':
    main()
