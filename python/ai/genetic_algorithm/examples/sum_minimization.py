import sys
import numpy as np
from .. import GeneticAlgorithm
from ../data_processing.binary_ops import bitsToBytes
sys.path.append('..')

def main():


    genesize = 10
    nbits = round(genesize/2)

    

    def fitness(gene):
        one = bitsToBytes(gene[:, 0:nbits])
        two = bitsToBytes(gene[:, nbits:])
        score = np.sum([one, two], axis=0)

        # print(one,two,score)
        return score

    print('nbits:', nbits)
    expected = 0
    # print('DesiredValue:', expected)

    ga = GeneticAlgorithm.GA(genesize, population_size=10,
                             epochs=1000, maximization=False)

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
        print('Achieved: ', score, 'Expected:', expected)
        
        return score
    print('BEST: ', best)
    evaluate(np.array([best]))
    # print(ga.history['statistics'])


if __name__ == '__main__':
    main()
