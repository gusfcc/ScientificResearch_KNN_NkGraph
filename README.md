# ScientificResearch_KNN_NkGraph
Modified K-Nearest Neighbours Algorithm (KNN) based on the Nk Interaction Graph.

Descripition: Source code of the tests made over the Modified KNN based on the nk Graph, made to test both K and alpha parameters, and compare with the original KNN.

Reference: 

Running the code:

  First, in a folder, separate the Datasets that are going to be used for the test. They need to have a 4 row header as below, followed by the dataset values separated by spaces, the last column being the label.
  
  MODEL: *STRING*  
  N_ATTRIBUTES: *INT*  
  N_OBJECTS: *INT*  
  DATASET:  
  *x1* *x2* *x3* ... *label*
  
  Then, change the parameters, located at the *main.py* file ([ ] are the default values):
  
    Ki - Start K parameter [01]
    Kf - Final K parameter [10]
      K is the number of neighbors, and the number of edges
        Ki >= 1
        Ki <= Kf

    Ai - Start A parameter [1]
    Af -  #Final A parameter [1]
      A is the number of edges defined by spatial density
      If Ai != Af, then Ki == Kf -> Tests are done separately on K and alpha
      (To run tests in both alpha and K simultaniously,
        it is necessary to change the alpha to vary accordingly to the varying K. This is not recomended because of runtime)
        Ai >= 0
          If A == 0 -> Modified KNN A equivalent to original KNN
        Ai <= Af
        Af <= Kf-1
        
    p  - crosval p-fold [10]
    DSFolder - Datasets folder [Datasets]
    createFiles - If all the steps of the creation of the graphs will be printed out [False]
    
    
    
At last, run *main.py*. The results will be printed in *Results.txt*
   

