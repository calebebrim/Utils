from ai.genetic_algorithm.GeneticAlgorithm import GA, GAU
from ai.data_processing.binary_ops import bitsNeededToNumber, bitsToBytes
import numpy as np


class ActivationFunctions(object):

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.power(np.e, -x))

    @staticmethod
    def relu(x):
        return (x>0)*x

    @staticmethod
    def tanh(x):
        return (2/(1+np.power(np.e, -2*x)))-1

    @staticmethod
    def binary_step(x):
        return 1 if x > 0 else 0

    @staticmethod
    def softplus(x):
        return np.log(1+np.power(np.e, x))
    
    @staticmethod
    def activations(name=False):
        if name:
            return ['ActivationFunctions.identity', 'ActivationFunctions.sigmoid', 'ActivationFunctions.tanh', 'ActivationFunctions.softplus', 'ActivationFunctions.relu']
        return [ActivationFunctions.identity, ActivationFunctions.sigmoid, ActivationFunctions.tanh, ActivationFunctions.softplus,ActivationFunctions.relu]
    

class GeneticNeuron(object):
    """
        The Objective here is optimize an classification 
        task with genetic algorithm.
    """

    def __init__(self, max_number=500,max_hidden_neurons=4, epochs=10000, ephoc_generations=100,population_size=10000, selection_count=5):
        self.max_number = max_number
        self.epochs = epochs
        self.ephoc_generations = ephoc_generations
        self.selection_count = selection_count
        self.population_size = population_size
        self.max_hidded_neurons = max_hidden_neurons

    def calculateNeurons(self,weights_matrix,):
        
        pass   
    
    def classify(self,genes=None,data=None):
        '''
            Usage: 

            (classification,error,activation,weights) = GA.classfy(genes=best_gene,data=[[...]])
            
        '''
        x = data[:, :-self.out_count]
        y = data[:, -self.out_count:]
        
        activations_bits_end_pos = self.activations_bits
        weights_power_bits_end_pos = activations_bits_end_pos + self.weigth_power_factor_bits
        bias_bits_end_pos = weights_power_bits_end_pos + self.bias_bits
        


        activation_bits = np.reshape(genes[0:activations_bits_end_pos],(self.weights_matrix_lines,-1))
        weights_power_bits = genes[activations_bits_end_pos:weights_power_bits_end_pos]
        bias_bits = genes[weights_power_bits_end_pos:bias_bits_end_pos]
        weigths_bits = genes[bias_bits_end_pos:]

        activation = bitsToBytes(activation_bits).reshape((self.weights_matrix_lines,-1))
        activation_limit = len(ActivationFunctions.activations())*np.ones(activation.shape)
        activation = np.min(np.concatenate((activation, activation_limit), axis=1), axis=1)
        # print("activation:", activation)
        weights_power = bitsToBytes(weights_power_bits)
        bias = bitsToBytes(np.reshape(bias_bits,(self.weights_matrix_lines,-1)))
        # print("bias:",bias)
        
        weights = bitsToBytes(np.reshape(weigths_bits, (self.weights_count, -1)))
        weights = weights.reshape((self.weights_matrix_lines,self.weights_matrix_columns))
        weights = weights - self.weight_subtractor
        weights = weights*np.power(10., -1*(weights_power))
        # weights = np.concatenate((weights, np.concatenate((
        #     np.zeros((self.in_count, self.weights_matrix_lines-self.in_count)),
        #     np.identity(self.in_count)),axis=1)
        # ), axis=0)
        # print("weights",weights.shape)
        # print(weights)


        # np.concatenate((np.zeros((20,4)),np.identity(4))) * np.ones((3,1,4)) * np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]]).reshape((3,1,-1))
        # data_reshape = x.reshape((x.shape[0], 1, -1))
        
            
        # last_wm_sector = data_reshape
        # print("last_wm:",last_wm_sector.shape)
        # print("weights_mateix",np.concatenate((weights,last_wm_sector),axis=1).shape)
        
        def w_column_to_neuron(column):
            return 1+self.out_count+column
        def w_column_to_w_row(column):
            return self.out_count+column
        def w_column_to_value_column(column):
            return column-self.max_hidded_neurons

        def neuron_to_w_column(neuron):
            return neuron-1-self.out_count

        def neuron_to_w_line(neuron):
            return neuron-1
        def line_pos_to_neuron(line):
            return line+1
        def line_pos_to_column(line):
            return line-self.out_count
        

        def process(weights,values,neuron_line,bias,activation,stack=""):
                if stack == "":
                    stack = "{}".format(line_pos_to_neuron(neuron_line))
                # print(stack)
                # neuron_line = neuron_to_w_line(neuron_line)
                
                neuron_line_w_pos_gt_zero = np.where(weights[neuron_line, :] != 0)[0]
                # print(line_pos_to_neuron(neuron_line), "neuron_line_w_pos_gt_zero",neuron_line_w_pos_gt_zero)
                
                
                available_positions = neuron_line_w_pos_gt_zero > line_pos_to_column(neuron_line)
                # print(line_pos_to_neuron(neuron_line),"available_positions:", available_positions, available_positions.shape)
                
                value_positions = \
                    neuron_line_w_pos_gt_zero[neuron_line_w_pos_gt_zero >= self.max_hidded_neurons]
                # print(line_pos_to_neuron(neuron_line),"value_positions:", value_positions, value_positions.shape)
                # print(line_pos_to_neuron(neuron_line),"value column:", w_column_to_value_column(value_positions), value_positions.shape)
                                        
                neurons_positions = \
                    neuron_line_w_pos_gt_zero[available_positions & (neuron_line_w_pos_gt_zero < self.max_hidded_neurons)]
                                        
                # print(line_pos_to_neuron(neuron_line),"neurons_positions:", neurons_positions,neurons_positions.shape)
                # print(line_pos_to_neuron(neuron_line), "neurons_row:", w_column_to_w_row(
                #     neurons_positions), neurons_positions.shape)
                # print(line_pos_to_neuron(neuron_line),"weights:", weights, weights.shape)
                # w = weights[neuron_line, neuron_line_w_pos_gt_zero]
                # w = w[forwarsd_neuron_w_pos_gt_zero]
                values_w = weights[neuron_line,value_positions]
                neuron_w = weights[neuron_line,neurons_positions]

                


                values_selector_in_values_matrix = value_positions-self.values_shift
                # print(line_pos_to_neuron(neuron_line),"values_selector_in_values_matrix",
                #       values_selector_in_values_matrix, values_selector_in_values_matrix.shape)
                # print("self.weights_matrix_lines", self.weights_matrix_lines)
                # print("self.in_count",self.in_count)
                # print("dependency neurons: ", neuron_line_w_pos_gt_zero[neurons_positions])
                # print("dependency values: ", neuron_line_w_pos_gt_zero[value_positions])
                # print("value index:", values_selector_in_values_matrix)
                
                # [print("{} > {}".format(line_pos_to_neuron(neuron_line), nxt,)) for nxt in w_column_to_neuron(neurons_positions)]
                # [print("{} * {}({})".format(line_pos_to_neuron(neuron_line), vi, vm))
                #  for vi, vm in zip(w_column_to_neuron(value_positions), w_column_to_value_column(value_positions))]
                
                # arr_val = np.sum(values_w * (values[:, :, values_selector_in_values_matrix]),axis=2)
                if(len(neurons_positions) > 0):
                    if len(neurons_positions)>1:
                        arr_neuron = tuple([process(weights=weights, values=values, neuron_line=n,bias=bias,activation=activation,stack="{}>{}".format(stack,line_pos_to_neuron(n))) for n in w_column_to_w_row(neurons_positions)])
                        arr_neuron = np.concatenate(arr_neuron,axis=1)
                        arr_neuron = arr_neuron.reshape((-1, len(neurons_positions)))
                        # print(line_pos_to_neuron(neuron_line), "arr_neuron",arr_neuron)
                    else: 
                        n =  w_column_to_w_row(neurons_positions)[0]
                        arr_neuron = process(weights=weights, values=values, neuron_line=n,bias=bias,activation=activation,stack="{}>{}".format(stack,line_pos_to_neuron(n)))
                        # print(line_pos_to_neuron(neuron_line), "arr_neuron",arr_neuron)
                    arr_final = np.dot( np.concatenate((arr_neuron, values[:,  values_selector_in_values_matrix]), axis=1), \
                                        np.concatenate((neuron_w, values_w), axis=0))
                else: 
                    arr_final = np.dot(values[:,  values_selector_in_values_matrix],values_w)    
                

                if len(arr_final.shape)==1:
                    arr_final = arr_final.reshape((-1,1))
                activation_index = min(len(ActivationFunctions.activations())-1,int(activation[neuron_line]))
                activation_fn = ActivationFunctions.activations()[activation_index]
                # print(activation_fn)
                return activation_fn(bias[neuron_line]+arr_final)
            
            
        classification = np.array([process(weights=weights, values=x,neuron_line=n,activation=activation,bias=bias) for n in range(self.out_count)]).reshape((-1,self.out_count))
        # print("classification")
        # print(classification.shape)
        
        error = classification-y
        abs_error = np.absolute(error)
        total_error = np.mean(np.power(abs_error,2))
        
        if total_error == np.NaN:
            total_error == np.inf 

        # raise Exception('should not pass!!!!')


        
       
        return classification,total_error,activation,weights

    def run(self, data, in_count):
        self.in_count = in_count
        self.out_count = data.shape[1]-self.in_count
        self.weigth_power_factor_bits = bitsNeededToNumber(5)
        self.weight_subtractor = self.max_number/2
        
        self.weights_matrix_lines = self.out_count+self.max_hidded_neurons
        self.bias_bits = self.weights_matrix_lines*bitsNeededToNumber(self.max_number)
        self.activations_bits =self.weights_matrix_lines*bitsNeededToNumber(len(ActivationFunctions.activations()))
        self.weights_matrix_columns = self.max_hidded_neurons+self.in_count
        self.weights_count = self.weights_matrix_lines \
                                * self.weights_matrix_columns
        self.weights_bits = self.weights_count \
                                * bitsNeededToNumber(self.max_number)

        self.nbits =  self.activations_bits \
                    + self.weigth_power_factor_bits \
                    + self.bias_bits \
                    + self.weights_bits

        self.ga = GA(gene_size=self.nbits,
                     population_size=self.population_size,
                     epochs=self.epochs,
                     ephoc_generations=self.ephoc_generations,
                     selection_count=self.selection_count)

        self.values_shift = self.max_hidded_neurons
        self.neuron_weight_position_shift = self.out_count
        print("""
            GeneticNeuron Start: 
                Number Of Inputs: {in_count}
                Number of bits for each gene: {nbits}
                Number of bits used for select activation func: {activation_bits}
                Number of classes: {n_classes} 
                Number of hidden neurons: {hidden_neurons}
                Weights matrix lines: {weights_matrix_lines} (out+hidden)
                Weights matrix columns: {weights_matrix_columns} (in+hidden)
                Values shift: {values_shift}
                weight shift: {weight_shift}
                
        """.format_map({ \
                "in_count": self.in_count,
                "nbits": self.nbits, \
                "activation_bits": self.activations_bits, \
                "n_classes": self.out_count,\
                "weights_matrix_lines": self.weights_matrix_lines, \
                "weights_matrix_columns": self.weights_matrix_columns, \
                "hidden_neurons": self.max_hidded_neurons,
                "values_shift":self.values_shift,
                "weight_shift":self.neuron_weight_position_shift}))

        

        if(self.out_count<=0):
            raise GeneticNeuronConfigurationError(
                """
                    The number of inputs({in_count}) on neuron does not match with data.shape. 
                    It must be greater than data.shape[1]. 
                    Perhaps you passed the datasite with wrong number of features ({features}).
                """.format(features=data.shape[1],in_count=self.in_count),errors=None)
        
        def fitness(genes):
            return self.classify(genes,data)[1] # 1 - for the error
        best_genes = self.ga.run(fitness, multiple=False)[0]
        self.best_genes =  best_genes
        



    

        


class GeneticNeuronConfigurationError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors





def main():
    import inspect
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # script filename (usually with path)
    # print(inspect.getfile(inspect.currentframe()))
    # script directory
    dataset_path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), os.pardir, os.pardir, os.pardir, os.pardir, 'datasets', 'iris'))
    
    
    iris_dataset = pd.read_csv(os.path.join(dataset_path, 'iris.csv'))
    species_dummies = pd.get_dummies(iris_dataset['species'], prefix='class')
    
    iris_dataset = pd.concat([iris_dataset, species_dummies], axis=1)
    iris_dataset.drop(columns=['species'], inplace=True)
    
    # print(iris_dataset.keys())
    # print(iris_dataset.head())

    iris_data = iris_dataset.values[:, 0:-3]
    iris_classes = iris_dataset.values[:, -2:]
    
    GN = GeneticNeuron(population_size=100, selection_count=5,max_hidden_neurons=10, max_number=20, epochs=1000000, ephoc_generations=10)
    
    X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_classes)

    GN.run(np.concatenate([X_train, y_train[:, :]], axis=1),in_count=4)
    
    
    data = np.concatenate([X_test, y_test[:,:]], axis=1)
    (classification, error, activation, weights) = GN.classify(
        genes=GN.best_genes, data=data)
    
    index1 = np.argsort(data[:, -1])
    index2 = np.argsort(data[:, -2])
    index3 = np.argsort(data[:, -3])

    from ai.data_processing.plotUtils import multi_plot
    from bokeh.layouts import column
    
    from bokeh.io import show

    plt1 = multi_plot([data[index1, -1] ,classification.T[2][index1].tolist()],legends=['original','previsto'],launch=False)
    
    plt2 = multi_plot([data[index2, -2] ,classification.T[1][index2].tolist()],legends=['original','previsto'],launch=False)
    
    plt3 = multi_plot([data[index3, -3] ,classification.T[0][index3].tolist()],legends=['original','previsto'],launch=False)
    
    show(column(plt1,plt2,plt3))
    

if __name__ == '__main__':
    main()
