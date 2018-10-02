import numpy as np


class GAU(object):
    '''
        Genetic Algorithm Util Functions

        __init_population__
        __mutation__
        __selection__
        __crossover__

    '''

    @staticmethod
    def __init_population__(gene_size, population_size, dtype=np.int, mn=-100000, mx=10000):
        '''  
            Must prepare for other types.
            currently only suport dtype bool
        '''

        
        if(dtype not in [np.bool, np.int
                         # TODO:
                         #   enable random population generation for dtypes:
                         #
                         #   np.float, np.doubl
                         #
                         ]):
            raise Exception('{} dtype not supported.'.format(dtype))

        if(dtype == np.bool):
            return np.random.choice([True, False], (population_size, gene_size))
        elif(dtype == np.int):
            return np.random.randint(mn, high=mx, size=(population_size, gene_size))

    @staticmethod
    def __mutation__(pop, mutation_prob=0.9, mn=-10000, mx=10000, dtype=np.bool):
        selector = np.random.choice([True, False], pop.shape, p=[mutation_prob, 1-mutation_prob])
        # print(selector.shape)
        if(dtype == np.bool):
            pop[selector] = np.invert(pop[selector])
        elif (dtype == np.int):
            pop[selector] = np.random.randint(
                mn, high=mx, shape=selector.sum())

        return pop

    @staticmethod
    def __selection__(pop, score, selection_count=4,maximization=False):
        
        if(maximization):
            sindex = (-score).argsort()
        else:
            sindex = score.argsort()
        # print(score[sindex]) # << score
        selection = pop[sindex[0:selection_count]]
        # selection = np.random.shuffle(selection)
        return selection, score[sindex]

    @staticmethod
    def __crossover__(pop):
        ''' 
            Crossover genes information of all samples
        '''
        # print(pop.shape)
        crosspoints = np.random.randint(
            1, high=max(2, pop.shape[1]-2), size=pop.shape[0])
        # print(crosspoints)

        for i in range(0, pop.shape[0]-2):
            cross = np.append(pop[i:pop.shape[0]-2, :crosspoints[i]],
                              pop[i+1:pop.shape[0]-1, crosspoints[i]:], axis=1)
            pop = np.append(pop, cross, axis=0)
        return pop


class GA():
    import numpy as np
    
    def __init__(self, gene_size, gene_type=np.bool, epochs=1000, selection_count=10, population_size=100,maximization = False, debug=False, verbose=True,ephoc_generations=100, population=GAU.__init_population__, mutation=GAU.__mutation__, crossover=GAU.__crossover__, selection=GAU.__selection__):
        ''' 
            Initialize Genetic Algorithm
            
            Default Usage: 
            
            from calebe import GeneticAlgorithm
            
            genesize = 10
            def bitsToBytes(values):
                return values.dot(2**np.arange(gene.shape[1])[::-1])
            fitness = lambda gene: sum(bitsToBytes(gene[:,:5]),bitsToBytes(gene[:,5:]))
            
            
            ga = GeneticAlgorithm.GA(genesize,population_size=100)
            best,pop,score = ga.run(fitness)
            

            
        '''
        # Documentation incomplete

        self.gene_size = gene_size
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.selection_count = selection_count
        self.population = population
        self.population_size = population_size
        self.populationType = gene_type
        self.maxepochs = epochs
        self.history = {
            "score": [],
            "population": [],
            "statistics": []
        }
        self.debug = debug
        self.verbose = verbose
        self.maximization = maximization
        self.best_score = 0 if maximization else 99999
        self.ephoc_generations = ephoc_generations
        if(verbose):
            print('''
                Generating Population With: 
                - Gene Size: {}
                - Population Size: {}
                - Population Type: {}
                - Ephocs: {}
                - Generations for each ephoch: {}
            '''.format(gene_size, population_size, gene_type,epochs,ephoc_generations))

    def run(self, fitness):
        def fitness_handler(pop,fitness):
            score = fitness(pop)
            evaluations = score.shape[0]
            samples = pop.shape[0]
            if  evaluations != samples:
                raise Exception("The number of returned evaluations ({}) must be equals to provided samples ({}). ".format(evaluations,samples))
            return score
        if(self.debug):print('Initializing Population...')
        pop = self.population(
            self.gene_size, self.population_size, dtype=self.populationType)
        

        score = fitness_handler(pop,fitness)
        statistics = self.statisics(pop, score)

        self.history['population'].append(pop)
        self.history['score'].append(score)
        self.history['statistics'].append(statistics)
        
        pop,score = self.selection(pop, score, self.selection_count,maximization=self.maximization)
        for i in range(1,self.maxepochs+1):
            
            
            
            pop = self.crossover(pop)
            if(self.debug):print('Crossover>>', pop.shape)
            pop = self.mutation(pop)
            if(self.debug):print('Mutation>>', pop.shape)
            score = fitness_handler(pop, fitness)
            statistics = self.statisics(pop, score)
            pop,score = self.selection(pop, score, selection_count=self.selection_count,maximization=self.maximization)
            if self.maximization:
                if(score[0] > self.best_score):
                    self.best_pop, self.best_score = (pop[0], score[0])
            else:
                if(score[0] < self.best_score):
                    self.best_pop, self.best_score = (pop[0], score[0])
            if(self.debug):print('Selection>>', pop.shape)
            # print(pop)
            if(self.verbose & ((i % self.ephoc_generations)==0)): 
                print("================EPOCH ({}/{})=======================".format(i, self.ephoc_generations))
                print(statistics)
                print('Best: ',self.best_score)

            self.history['population'].append(pop)
            self.history['score'].append(score)
            self.history['statistics'].append(statistics)
        # pop = self.selection(pop, score, self.selection_count)
        
        return self.best_pop, pop, score

    def statisics(self, pop, score):
        '''
            Calculate statistics of each individual and save the scores
        '''
        metrics = {'max': np.max(score), 'min': np.min(
            score), 'avg': np.average(score)}
        # print('Max: ', np.max(score), ' Min: ', np.min(score), ' Average: ', np.average(score))
        # print(metrics)
        return metrics


def main():
    import numpy as np
    from calebe import GeneticAlgorithm
    genesize = 10
    nbits = round(genesize/2)

    
    def bitsToBytes(values):
        processed = np.array(values.dot(2**np.arange(values.shape[1])[::-1]))
        return processed
    
    def fitness(gene):
        one = bitsToBytes(gene[:, 0:nbits])
        two = bitsToBytes(gene[:, nbits:])
        score = np.sum([one, two], axis=0)
                        
        # print(one,two,score)
        return score
    
    print('nbits:',nbits)
    expected = bitsToBytes(np.array([[True]*5])) + \
        bitsToBytes(np.array([[True]*5]))
    # print('DesiredValue:', expected)
    



    ga = GeneticAlgorithm.GA(genesize, population_size=10,epochs=1000,maximization=False)

    ga.debug = False
    ga.verbose = True
    
    best, pop, score = ga.run(fitness)
    # print(score)

    def evaluate(gene):
        print('==========Evaluation=============')
        one_bits = gene[:, 0:nbits]
        # print(one_bits.shape)
        one = bitsToBytes(one_bits)
        two = bitsToBytes(gene[:, nbits:])
        score = np.sum([one, two], axis=0)
        
        # print(one,two,score)
        print('Achieved: ',score,'Expected:',expected)
        return score
    print('BEST: ',best)
    evaluate(np.array([best]))
    # print(ga.history['statistics'])

if __name__ == '__main__':
    main()
