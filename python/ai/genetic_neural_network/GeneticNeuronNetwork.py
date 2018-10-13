from ai.genetic_algorithm.GeneticAlgorithm import GA,GAU
from ai.data_processing.binary_ops import bitsNeededToNumber,bitsToBytes
import numpy as np


class ActivationFunctions(object):

    @staticmethod
    def identity(x):
        return x
    @staticmethod
    def sigmoid(x):
        return 1/1+np.power(np.e, -x)

    @staticmethod
    def relu(x):
        return np.apply_along_axis(lambda v: v if v>0 else 0,axis=0,arr=x)

    @staticmethod
    def tanh(x):
        return (2/(1+np.power(np.e,-2*x)))-1
    
    @staticmethod
    def binary_step(x):
        return 1 if x>0 else 0
    
    @staticmethod
    def softplus(x):
        return np.log(1+np.power(np.e,x))

    @staticmethod
    def activations():
        return [ActivationFunctions.identity, ActivationFunctions.sigmoid,ActivationFunctions.tanh,ActivationFunctions.softplus]


class GeneticNeuron(object):
    """
        The Objective here is optimize an classification 
        task with genetic algorithm.
    """
    


    def __init__(self,in_count,max_number=100):
        self.in_count = in_count
        self.activations_bits = bitsNeededToNumber(len(ActivationFunctions.activations()))
        self.nbits = (in_count*bitsNeededToNumber(max_number))+self.activations_bits
        self.weight_subtractor = max_number/2
        self.ga = GA(gene_size=self.nbits,population_size=1000,maximization=False,epochs=50,ephoc_generations=10)

        print("""
            GeneticNeuron Start: 
                Number Of Inputs: {in_count}
                ...
        """.format_map({"in_count":self.in_count}))
    
    def run(self,data,forward_penality):
        out_count = data.shape[1]-self.in_count
        if(out_count<=0):
            raise GeneticNeuronConfigurationError("The number of inputs({in_count}) on neuron does not match with data.shape. It must be greater than data.shape[1]. Perhaps you passed the datasite with wrong number of features ({features}) ".format(features=data.shape[1],in_count=self.in_count),errors=None)
        def optimize(genes,data):
            activation_bits = genes[0:self.activations_bits]
            weigths_bits = genes[self.activations_bits:]
            activation = min([bitsToBytes(activation_bits),len(ActivationFunctions.activations())-1])
            # print(activation)
            weights = bitsToBytes(np.reshape(weigths_bits, (self.in_count, -1)))
            weights = (weights - self.weight_subtractor)*0.1
            
            DP = np.dot(data[:,:-out_count],weights.T)
            classification = ActivationFunctions.activations()[activation](DP)
            error = np.mean(np.abs(data[:,-1] - classification))

            return error,weights,activation,classification
        def fitness(genes):
            return optimize(genes,data)[0]
        (best_genes, pop, score) = self.ga.run(fitness, multiple=False)
        
        (error,weights,activation,classification) = optimize(best_genes,data)
        
        
        
        
        import matplotlib.pyplot as plt
        index = np.argsort(data[:,-1])
        print(weights)
        # print(classification)
        # print(data[:,-1])
        # print(index)

        print('Activation: ',activation)
        
        x = range(0,len(index))
        
        plt.plot(x, data[index, -1])
        plt.plot(x, classification[index])
        plt.show()








class GeneticNeuronConfigurationError(Exception):
    def __init__(self,message,errors):
        super().__init__(message)
        self.errors = errors


def testInCount():
    try:
        GN = GeneticNeuron(in_count=4,out_count=3, max_number=10)
        GN.run(np.array([[]]),None)
    except Exception as ex:
        assert("GeneticNeuronConfigurationError", ex.__class__.__name__)

def main():
    import inspect
    import os
    import pandas as pd
    # script filename (usually with path)
    print(inspect.getfile(inspect.currentframe()))
    # script directory
    dataset_path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), os.pardir, os.pardir, os.pardir, os.pardir,'datasets','iris'))
    iris_dataset = pd.read_csv(os.path.join(dataset_path, 'iris.csv'))
    iris_dataset = pd.concat([iris_dataset,pd.get_dummies(iris_dataset['species'],prefix='class')],axis=1)
    iris_dataset.drop(columns=['species'],inplace=True)
    print(iris_dataset.keys())
    print(iris_dataset.head())

    iris_data = iris_dataset.values[:, 0:-3]
    iris_classes = iris_dataset.values[:, -3:]
    GN = GeneticNeuron(in_count=4, max_number=10)

    GN.run(np.concatenate([iris_data,iris_classes[:,[0]]],axis=1),None)

if __name__ == '__main__':
    main()
