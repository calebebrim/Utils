import numpy as np
from collections import Counter

def KNNClassifier(data,predict,K=3):
    if len(data)>K:
        raise('not enought data to classify')
    distances = []
    for group in data: 
        for features in data[group]:
            distances.append([euclidean_distance(features,predict),group])
    votes = [i[1] for i in sorted(distances)[:K]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


    
def euclidean_distance(data,comparable):
    return np.linalg.norm(np.array(data)-np.array(comparable))



def main():
    data = {1:[[1,2,3],[1,2,1],[1,3,4]],2:[[10,20,10],[30,10,30],[40,30,10]]}
    print('class',KNNClassifier(data,[1,2,3]))
    print('class', KNNClassifier(data, [10, 20, 30]))
    print('class', KNNClassifier(data, [1, 2, 30]))

    print(euclidean_distance([1, 2, 3], [1, 10, 5]))
    print(euclidean_distance([1, 2, 3], [1, 0, 5]))
    

if __name__ == '__main__':
    main()
