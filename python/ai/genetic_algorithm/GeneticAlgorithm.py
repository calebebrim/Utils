# Author: Calebe Brim
# Date: 02/08/18
import numpy as np


class GAU(object):
    '''
        Genetic Algorithm Util Functions

        __init_population__
        __mutation__
        __selection__
        __crossover__
        __statisics__

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
    def __mutation__(pop, mutation_prob=0.6, mn=-10000, mx=10000, dtype=np.bool):
        selector = np.random.choice([True, False], pop.shape, p=[mutation_prob, 1-mutation_prob])
        # print(selector.shape)
        # print(selector)
        if(dtype == np.bool):
            pop[selector] = np.invert(pop[selector])
        elif (dtype == np.int):
            pop[selector] = np.random.randint(mn, high=mx, shape=selector.sum())

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

    @staticmethod
    def __statisics__(pop, score):
        '''
            Calculate statistics of each individual and save the scores
        '''
        metrics = {'max': np.max(score), 'min': np.min(
            score), 'avg': np.average(score)}
        # print('Max: ', np.max(score), ' Min: ', np.min(score), ' Average: ', np.average(score))
        # print(metrics)
        return metrics

class GA():
    import numpy as np
    
    def __init__(self, gene_size, gene_type=np.bool, epochs=1000, selection_count=10, population_size=100,maximization = False, debug=False, verbose=True,ephoc_generations=100, population=GAU.__init_population__, mutation=GAU.__mutation__, crossover=GAU.__crossover__, selection=GAU.__selection__,statistics=GAU.__statisics__,on_ephoc_ends=None):
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
        self.statisics = statistics
        self.on_ephoc_ends_callback = on_ephoc_ends
        if(verbose):
            print('''
                Generating Population With: 
                - Gene Size: {}
                - Population Size: {}
                - Population Type: {}
                - Ephocs: {}
                - Generations for each ephoch: {}
            '''.format(gene_size, population_size, gene_type,epochs,ephoc_generations))

    def on_ephoc_ends(self,pop,score,statistics):
        self.history['population'].append(pop)
        self.history['score'].append(score)
        self.history['statistics'].append(statistics)
        if self.on_ephoc_ends_callback:
            self.on_ephoc_ends_callback()
    
    def fitness_handler(self, pop, fitness,paralel,threads=1,multiple=True):
        if paralel:
            from concurrent.futures import ThreadPoolExecutor

            # def executor(individual,index,scores):
            #     scores[index] = fitness(individual)

            def executor_fn(index):
                return fitness(pop[index])

            e = list(range(pop.shape[0]))
            score = np.empty(pop.shape[0])

            with ThreadPoolExecutor(max_workers=threads) as executor:

                # print(future.result())
                for p in range(pop.shape[0]):
                    e[p] = executor.submit(executor_fn, p)
                # ev = threading.Thread(target=executor, args=(pop[p],p,score))
                # ev.start()
                # e.append(ev)

                for p in range(pop.shape[0]):
                    score[p] = e[p].result()
            score = np.array(score)
        elif multiple:
            # print('Using Multiple')
            score = fitness(pop)
        else:
            # print('Not Using Multiple')
            score = []
            for gene in pop:
                curr = fitness(gene)
                score.append(curr)
            score = np.array(score)

        evaluations = score.shape[0]
        samples = pop.shape[0]

        if evaluations != samples:
            raise Exception("The number of returned evaluations ({}) must be equals to provided samples ({}). ".format(
                evaluations, samples))
        return score



    def run(self, fitness,paralel=False,threads=1,multiple=True):
        '''
            Used to effectivelly run the genetic algorithm

            Usage: 
            
                ga = GA(gene_size=1000,population_size=100,maximization=False,epochs=100)

                (best_pop, pop, score) = ga.run(lambda genes: sum(genes),multiple=False)
        '''
        if(self.debug):print('Initializing Population...')
        pop = self.population(
            self.gene_size, self.population_size, dtype=self.populationType)
        
        score = self.fitness_handler(pop,fitness=fitness,paralel=paralel,threads=threads,multiple=multiple)
        statistics = self.statisics(pop, score)
        pop,score = self.selection(pop, score, self.selection_count,maximization=self.maximization)

        self.on_ephoc_ends(pop, score, statistics)
        for i in range(1,self.maxepochs+1):
            if(self.debug):print('epoch>>', i)
            
            pop = self.crossover(pop)
            # if(self.debug):print('Crossover>>', pop.shape)
            pop = self.mutation(pop)
            # if(self.debug):print('Mutation>>', pop.shape)
            
            score = self.fitness_handler(pop, fitness=fitness, paralel=paralel, threads=threads, multiple=multiple)

            statistics = self.statisics(pop, score)
            pop,score = self.selection(pop, score, selection_count=self.selection_count,maximization=self.maximization)
            
            if self.maximization:
                if(score[0] > self.best_score):
                    self.best_pop, self.best_score = (pop[0], score[0])
            else:
                if(score[0] < self.best_score):
                    self.best_pop, self.best_score = (pop[0], score[0])


            if(self.debug):print('Selection>>', pop.shape)
            

            if(self.verbose & ((i % self.ephoc_generations)==0)): 
                print("================EPOCH ({}/{})=======================".format(i, self.ephoc_generations))
                print(statistics)
                print('Best: ',self.best_score)

            self.on_ephoc_ends(pop, score, statistics)
        
        return self.best_pop, pop, score

    

