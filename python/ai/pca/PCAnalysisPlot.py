


import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PCAnalysisPlot(object):

    def __init__(self,DATA):
        if type(DATA) == str: 
            self.data = pd.read_csv(DATA)
        else:
            self.data = DATA

        self.pca = PCA(n_components=2)


        self.principalC = self.pca.fit_transform(self.data[self.data.columns[self.data.dtypes==float]])
        self.principalDf = pd.DataFrame(data=self.principalC, columns=[
                                   'principal component 1', 'principal component 2'])
        

    def plot(self,target_column = None,title='Principal Components Analysis'):
        if target_column != None: 
            
            
            finalDf = pd.concat([self.principalDf, self.data[[target_column]]], axis=1)

            fig = plt.figure(figsize=(8, 8))

            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_title(title, fontsize=20)
            targets = finalDf[target_column].unique()
            colors = ['r', 'g', 'b']
            for target, color in zip(targets, colors):
                indicesToKeep = finalDf[target_column] == target
                ax.scatter(
                    finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
            ax.legend(targets)
            ax.grid()
        else:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_title(title, fontsize=20)

            ax.scatter(
                self.principalDf['principal component 1'],
                self.principalDf['principal component 2'], c='b', s=50)

            # ax.legend(targets)
            ax.grid()
        plt.show()

def main():
    pca = PCAnalysisPlot(
        '/Users/calebebrim/anaconda/envs/py36/lib/python3.6/calebe/datasets/iris/iris.csv')
    
    pca.plot('species')
    pca.plot(title='Distrib')


if __name__ == '__main__':
    main()
