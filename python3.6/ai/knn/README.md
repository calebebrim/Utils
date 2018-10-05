# KNN Experiments

Simple Implementation of KNN algorithm
Supported python version = python3.6



## Usage:
### 1 - Classification:
    from knn import KNNClassifier
    data = {1:[[1,2,3],[1,2,1],[1,3,4]],2:[[10,20,10],[30,10,30],[40,30,10]]}
    print('class',KNNClassifier(data,[1,2,3]))
    print('class', KNNClassifier(data, [10, 20, 30]))
    print('class', KNNClassifier(data, [1, 2, 30]))
### 2 - Getting the euclidean distance
    from knn import euclidean_distance as eud
    
    print(eud([1,2,3],[1,20,3]))
    print(eud([1,2,3],[1,10,3]))