import numpy as np
from numpy.random import seed
from numpy.random import randint
import matplotlib.pyplot as plt
import functools
import operator
import matplotlib
import itertools
import random
import os
import itertools

# global variables
qualities = []
num_parents_mating = 4
# Mutation percentage
mutation_percent = 0.9

sol_per_pop = 8

face=[
        0,0,0,0,0,0,0,0,0,0,
        0,1,1,1,1,1,1,1,1,0,
        0,1,0,0,0,0,0,0,1,0,
        0,1,0,1,0,0,1,0,1,0,
        0,1,0,0,0,0,0,0,1,0,
        0,1,0,1,0,0,1,0,1,0,
        0,1,0,1,1,1,1,0,1,0,
        0,1,0,0,0,0,0,0,1,0,
        0,1,1,1,1,1,1,1,1,0,
        0,0,0,0,0,0,0,0,0,0
    ]

faceVector = np.array(face)
face = faceVector.reshape((10, 10)).astype("uint8")
imgplot = plt.imshow(face)


# convert chromosom to matrix
def convert2img(chromo, imgShape):
    imgArr = np.reshape(a=chromo, newshape=imgShape)
    return imgArr


# fitness function
def fitness_fun(target_chrom, indiv_chrom):
    error = -1 * np.sum(np.abs(indiv_chrom - target_chrom))
    return error


# fitnes values of all chrosomes
def cal_pop_fitness(target_chrom, pop):
    qualities = np.zeros(pop.shape[0])
    for indv_num in range(pop.shape[0]):
        qualities[indv_num] = fitness_fun(target_chrom, pop[indv_num, :])
    return qualities


# generate some integers
# values = randint(0, 2, 100)
def initialPopulation(imgShape, nIndividuals=8):
    seed(1)
    init_population = np.empty(
        shape=(nIndividuals, functools.reduce(operator.mul, imgShape)), dtype=np.uint8
    )
    for indv_num in range(nIndividuals):
        # Randomly generating initial population chromosomes genes values.
        init_population[indv_num, :] = randint(0, 2, 100)
    return init_population


def getBestParents(population, qualities, n):
    parents = []
    indx = np.argsort(qualities)[::-1]
    n_fit = np.sort(qualities)[::-1]
    print("Index of best-{}".format(indx))
    print("Best Qualities-{}".format(n_fit))
    # take best 4 parents
    best_parent_indx = indx[0:n]
    for i in best_parent_indx:
        #         print(i)
        parents.append(population[i])
    # qualities[i] = -1

    parents = np.array(parents)
    return parents


def crossover(parents, img_shape, n_individuals=8):
    new_population = np.empty(
        shape=(n_individuals, functools.reduce(operator.mul, img_shape)), dtype=np.uint8
    )

    new_population[0 : parents.shape[0], :] = parents

    # Getting how many offspring to be generated. If the population size is 8 and number of parents mating is 4,
    # then number of offspring to be generated is 4.
    num_newly_generated = n_individuals - parents.shape[0]
    # Getting all possible permutations of the selected parents.
    parents_permutations = list(
        itertools.permutations(iterable=np.arange(0, parents.shape[0]), r=2)
    )
    # Randomly selecting the parents permutations to generate the offspring.
    selected_permutations = random.sample(
        range(len(parents_permutations)), num_newly_generated
    )

    comb_idx = parents.shape[0]
    for comb in range(len(selected_permutations)):
        # Generating the offspring using the permutations previously selected randmly.
        selected_comb_idx = selected_permutations[comb]
        selected_comb = parents_permutations[selected_comb_idx]

        # Applying crossover by exchanging half of the genes between two parents.
        half_size = np.int32(new_population.shape[1] / 2)
        new_population[comb_idx + comb, 0:half_size] = parents[
            selected_comb[0], 0:half_size
        ]
        new_population[comb_idx + comb, half_size:] = parents[
            selected_comb[1], half_size:
        ]

    return new_population


def mutation(population, numParentsMating, mutPercent):

    for idx in range(numParentsMating, population.shape[0]):

        # Selecting specific percentage of genes randomly

        rand_idx = np.uint32(
            np.random.random(size=np.uint32(mutPercent / 100 * population.shape[1]))
            * population.shape[1]
        )

        # Genes are selected at a random an their values are changed

        new_values = np.uint8(np.random.random(size=rand_idx.shape[0]) * 256)

        # Updating the population .

        population[idx, rand_idx] = new_values

    return population


population = initialPopulation((10, 10))


def save_images(currIteration, qualities, population, imShape, save_point, save_dir):
    if np.mod(currIteration, save_point) == 0:
        # Choosing best solution chromosome
        best_solution_chrom = population[
            np.where(qualities == np.max(qualities))[0][0], :
        ]
        # Converting the matrix into image.
        best_solution_img = convert2img(best_solution_chrom, imShape)
        # Saving image in a directory.
        matplotlib.pyplot.imsave(
            save_dir + "solution_" + str(currIteration) + ".png", best_solution_img
        )


for iteration in range(10000):
    qualities = cal_pop_fitness(faceVector, population)
    print("Quality : ", qualities, ", Iteration : ", iteration)
    parents = getBestParents(population, qualities, num_parents_mating)
    population = crossover(parents, (10, 10), n_individuals=sol_per_pop)
    population = mutation(
        population,
        num_parents_mating,
        mutation_percent,
    )
    save_images(
        iteration,
        qualities,
        population,
        (10, 10),
        save_point=1000,
        save_dir=os.curdir + "/output/",
    )
