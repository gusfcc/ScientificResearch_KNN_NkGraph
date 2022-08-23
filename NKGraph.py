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

class NKGraph:
    
    def __init__(self, name):
        self.name = name
        
        self.readDataSet()        
        self.distances = pd.DataFrame(squareform(pdist(self.dt.iloc[:,:],'sqeuclidean')),columns = self.dt.index, index = self.dt.index)
        self.density = self.calc_density(self.distances)
    
    #########################################
    
    def readDataSet(self):
        df = pd.read_table("DataSets/" + str(self.name) + ".dat", sep = '\s+', header = None, skiprows = 4, index_col = False)
        df.name = self.name
        
        self.dt = df.iloc[:,:len(df.columns)-1].copy() #Data
        #self.dt = (self.dt - self.dt.min()) / (self.dt.max() - self.dt.min())
        self.dt.name = self.name
        self.dtl = df.iloc[:,len(df.columns)-1].copy() #Label
    
    #########################################
    
    def calc_density(self, distances):
        percent = 2
        vD = np.sort(np.triu(distances.values).flatten())
        vD = vD[vD != 0]
        pos = int(len(vD)* percent/100)
        dc = vD[pos]
        pdLen = distances.columns.tolist()
        
        ndaDen = pd.DataFrame(0, index = pdLen, columns = ["Den"], dtype = float)
        
        for i, col in distances.iterrows():
            rho = 0
            for j in col:
                rho = rho + m.exp(-(j/dc)**2)
            
            ndaDen.at[i, "Den"] = rho
      
        return ndaDen

    #########################################

    def createNKGraph(self, K, alpha, distances, dt, density):
        DG = nx.DiGraph()
        n = len(dt.index)
        dist2 = distances.copy()
        maxD = dist2.values.max() + 1
        
        
        for x, col in dt.iterrows():
            nodex = col.name
            DG.add_node(nodex)
            DG.add_edge(nodex, nodex)
            dist2.at[x,x] = maxD 
        
        if (K >= 2):            
            Kdist = (int) (np.ceil(((1-alpha) * (K))))
            Kdist = Kdist - 1
            Kdens = K - Kdist - 1
            
            for i, col in dt.iterrows():                
                density2 = density.copy()
                for k in range(Kdens):
                    delta = maxD
                    for j,col2 in dt.iterrows():
                        if(density2.at[i,"Den"] < density2.at[j,"Den"]):
                            if(dist2.at[i,j] < delta):
                                delta = dist2.at[i,j]
                                j_min = j
                    
                    if(delta == maxD):
                        if(i > dist2.index[0]):
                            j_min = dist2.index[0]
                        else:
                            j_min = dist2.index[1]
                        for j,col2 in dt.iterrows():
                            if(j != i and dist2.at[i,j_min] > dist2.at[i,j] and density2.at[j,"Den"] != -1):
                                j_min = j
                                
                    nodex = col.name
                    nodey = j_min
                    DG.add_edge(nodex,nodey)
                    dist2.at[j_min,i] = maxD
                    density2.at[j_min,"Den"] = -1
                    
            for x, col in dt.iterrows():
                nodex = col.name
                Ksorted = np.argsort(dist2[x].to_numpy()).reshape(n)
                KNN = Ksorted[0:Kdist]
                for y in KNN:
                    nodey = dist2.iloc[y].name
                    DG.add_edge(nodex,nodey)
                
        return DG
    
    #########################################
    
    def KNN(self, x, K, distances, dt, dtl, createFiles = True, fPath = ''):
           
        
        vx = cdist(np.array([x]), dt.to_numpy(), 'sqeuclidean').argmin()
        
        n = len(dt.index)

        Ksorted = np.argsort(distances.iloc[[vx],:].to_numpy()).reshape(n)
        KNN = list(distances.iloc[Ksorted[0:K]].index)
        
        unique, pos = np.unique(dtl.iloc[KNN],return_inverse = True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        c = unique[maxpos]
        
        if(createFiles is True):
        
            fName = fPath + '/' + self.name + '_' + str(K) + "_KNN.txt"
            file = open(fName,"a")    
        
            file.write("x index       : " + str(x.name) + "\n")
            file.write("x             : " + str(list(x)) + "\n")
            file.write('\n')
            file.write("vx index      : " + str(dt.iloc[vx].name) + "\n")
            file.write("vx            : " + str(list(dt.iloc[vx])) + "\n")
            file.write('\n')
            file.write("KNN           : " + str(list(KNN)))
            file.write("KNN labels    : " + str(list(dtl.iloc[KNN])) + "\n")
            file.write('\n')
            file.write("x Label       : " + str(dtl.iloc[int(x.name),]) + "\n")
            file.write("Classification: " + str(c) + "\n")
            file.write('\n\n') 
            
            file.close()
        
        return c
    
    #########################################
    
    def KNNCrossValKFold(self, K, p, createFiles = True, fPath = ''):
        kf = KFold(p)
        count = 0
        all = 0
        
        if(createFiles is True):
            FName = fPath + '/' + self.name + '_' + str(K) + "_KNN.txt"
            file = open(FName,"a")    
        
        dtl_pred = np.zeros(np.size(self.dtl))
        
        for train, test in kf.split(self.dt):
        
            tD = self.distances.loc[train,train]
            tDt = self.dt.loc[train,]
            tDtl = self.dtl.loc[train,]       
            tDt.name = self.dt.name
                        
            for i in test:       
                dtl_pred[i] = self.KNN(self.dt.loc[i], K, tD, tDt, self.dtl, createFiles = createFiles,fPath = fPath) 
                
        confM = CM(self.dtl, dtl_pred)
        
        if(createFiles is True):
            FName = fPath + '/' + self.name + '_' + str(K) + "_KNN.txt"
            file = open(FName,"a")
            file.write("\n")
            file.write(str(confM))
            file.write("\n")
            file.close()
        
        return confM
    
    #########################################
    
    def nkAclassifier(self, x, K, alpha, dt, dtl, nkgraph, createFiles = True, fPath = ''):
        vx = cdist(np.array([x]), dt.to_numpy(), 'sqeuclidean').argmin()
        
        vxAdj = list(nkgraph.adj[dt.iloc[vx].name].keys())
        unique, pos = np.unique(dtl.loc[vxAdj],return_inverse = True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        c = unique[maxpos]
        
        if(createFiles is True):
        
            FName = fPath + '/' + self.name + '_' + str(K) + '_' + str((int)(alpha*K)) + "_nkA.txt"
            file = open(FName,"a")    
            
            file.write("x index       : " + str(x.name) + "\n")
            file.write("x             : " + str(list(x)) + "\n")
            file.write('\n')
            file.write("vx index      : " + str(dt.iloc[vx].name) + "\n")
            file.write("vx            : " + str(list(dt.iloc[vx])) + "\n")
            file.write('\n')
            file.write("vx Adj        : " + str(list(vxAdj)) + "\n")
            file.write("vx Adj Labels : " + str(list(dtl.iloc[vxAdj])) + "\n")
            file.write("x Label       : " + str(dtl.iloc[int(x.name),]) + "\n")
            file.write('\n')
            file.write("Classification: " + str(c) + "\n")
            file.write('\n\n')
            
            file.close()
        
        return c
    
    #########################################
    
    def nkBclassifier(self, x, K, alpha, dt, dtl, nkgraph, createFiles = True, fPath = ''):
        neighLabels = cdist(np.array([x]), dt.to_numpy(), 'sqeuclidean').argsort()[0]
        neighborsX = np.array(dt.iloc[neighLabels].index)    
        vx = neighborsX[0]    
        neighborsX[neighborsX == vx] = neighborsX.size * 2
        vj = np.zeros(K, int)
        vj[0] = vx
        
        
        for i in range(1,K):
            vxAdj = list(nkgraph.adj[dt.loc[vx].name].keys())
            intersect, iAdj, iNeighX = np.intersect1d(vxAdj, neighborsX, return_indices = True)
            vj[i] = neighborsX[np.sort(iNeighX)[0]] 
            vx = vj[i]        
            neighborsX[neighborsX == vx] = neighborsX.size * 2
                   
            
        unique, pos = np.unique(dtl.loc[vj],return_inverse = True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        c = unique[maxpos]
        
        if(createFiles is True):
        
            FName = fPath + '/' + self.name + '_' + str(K) + '_' + str((int)(alpha*K)) + "_nkB.txt"
            file = open(FName,"a")     
            
            file.write("x index       : " + str(x.name) + "\n")
            file.write("x             : " + str(list(x)) + "\n")
            file.write('\n')
            file.write("1st vx index  : " + str(dt.loc[vj[0]].name) + "\n")
            file.write("1st vx        : " + str(list(dt.loc[vj[0]])) + "\n")
            file.write('\n')
            file.write("vj list       : " + str(list(vj)) + "\n")
            file.write("vj list Labels: " + str(list(dtl.loc[vj])) + "\n")
            file.write("x Label       : " + str(dtl.iloc[int(x.name),]) + "\n")
            file.write('\n')
            file.write("Classification: " + str(c) + "\n")
            file.write('\n\n')
                
            file.close()
        
        return c
    
    #########################################
    
    def nkCrossValKFold(self, K, alpha, p, type = 'b', createFiles = True, fPath = ''):
        kf = KFold(p)
        count = 0
        all = 0
        
        if(createFiles is True):
            FName = fPath + '/' + self.name + '_' + str(K) + '_' + str((int)(alpha*K)) + "_nk" + str(type).upper() + ".txt"
            file = open(FName,"w").close()
        
        dtl_pred = np.zeros(np.size(self.dtl))
                    
        for train, test in kf.split(self.dt):
            
            tD = self.distances.loc[train,train]       
            tDt = self.dt.loc[train,]
            tDtl = self.dtl.loc[train,]
            tG = self.createNKGraph(K, alpha, tD, tDt, self.calc_density(tD))
                        
            tDt.name = self.dt.name
                        
            for i in test:
                if type == 'a':                
                    dtl_pred[i] = self.nkAclassifier(self.dt.loc[i], K, alpha, tDt, self.dtl, tG, createFiles = createFiles, fPath = fPath)
                elif type == 'b':            
                    dtl_pred[i] = self.nkBclassifier(self.dt.loc[i], K, alpha, tDt, self.dtl, tG, createFiles = createFiles, fPath = fPath)
                else: return 0
                            
        confM = CM(self.dtl, dtl_pred)  
        
        if(createFiles is True):
            FName = fPath + '/' + self.name + '_' + str(K) + '_' + str((int)(alpha*K)) + "_nk" + str(type).upper() + ".txt"
            file = open(FName,"a")
            file.write("\n")
            file.write(str(confM))
            file.write("\n")
            file.close()
        
        return confM
    
    #########################################
    
    def graphAdjFile(self, nkgraph, fPath = "NK.txt"):
        file = open(fPath,"w")
        
        for i in nkgraph.adjacency():
            adj = list(i[1].keys())
            for j in adj:
                file.write(str(j)+ " ")
            file.write('\n')
        
        file.close()
    
    #########################################
    
    def KNNFile(self, KNN, fPath = "KNN.txt"):
        file = open(fPath,"a")
        
        for i in KNN:
            file.write(str(i)+ " ")
        file.write('\n')
        
        file.close()
    
    #########################################
    
    def distFile(self, distances, fPath = "dist.txt"):
        file = open(fPath,"w")
        
        pd.set_option('display.max_rows', distances.shape[0]+1)
        for i, col in distances.iterrows():
            file.write(str(i) + "\n" + str(col) + "\n")
            file.write('\n\n\n')
        
        file.close()
    
    #########################################
    
    def drawNKGraph(self, nkgraph, dt):
        pos = dt.to_dict("index")  
        for i in pos:
            pos[i] = list(pos[i].values())    
        
        nx.draw_networkx(nkgraph, pos)
        plt.show()
        
        #print(nkgraph.adj)