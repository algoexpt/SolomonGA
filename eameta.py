# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:25:46 2021

@author: Ajay
"""

import os.path
from solomon import Solomon
from deap import base, tools, creator, algorithms
import random
import array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sat
import elitism
import math

PROBLEM = [("C101", 10), ("C201", 3), ("R101", 19), ("R201", 4), ("RC105", 13)]
PN = PROBLEM[4]
s = Solomon(PN[0], number=PN[1])

random.seed(64)

if "FitnessMin" not in dir(creator):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if "Individual" not in dir(creator):
    creator.create("Individual", array.array, typecode='i',
                   fitness=creator.FitnessMin)

# ---------------------------------------------------------------------------
# EA params


class EAMeta:
    def __init__(self):
        self.TOURNAMENT_SIZE = 2
        self.POPULATION_SIZE = 100
        self.P_CROSSOVER = 0.8
        self.P_MUTATION = 0.2
        self.MAX_GENERATIONS = 100
        self.HALL_OF_FAME_SIZE = 10

        # meta params
        self.NBR_DIST = 10
        self.META_SIZE = 2
        self.META_ITERATIONS = 500

    def initForIteration(self, i):
        j = int(i/10)
        self.META_SIZE = 3 + j
        self.MAX_GENERATIONS = 30 + (10 * (j+1))
        self.P_MUTATION = 0.1 + j / 100


eam = EAMeta()

# ---------------------------------------------------------------------------
# algo = elitism.eaSimpleWithElitism
# algo = algorithms.eaSimple
# ---------------------------------------------------------------------------


def evaluateCandidate(candidate):
    # 1. collect locations per vehicle
    trucks = s.getSolutionFromChromosome(candidate)

    # 2. find scores
    total_score = 0
    for t in trucks:
        t.sort()
        tkey = tuple(t)
        if tkey not in s.sequenced:
            seqt = sat.sequence(s, t)
            s.sequenced[tkey] = s.calcTruckScore(seqt)

        total_score += s.sequenced[tkey]

    return total_score,


def createCopy(initialsol):
    return initialsol


def createRandom(initialsol):
    nr_trucks = max(initialsol) + 1
    nr_locations = len(initialsol)
    choice = random.choices(population=range(nr_trucks), k=nr_locations)
    return choice


def createUniform(initialsol):
    nr_trucks = max(initialsol) + 1
    nr_locations = len(initialsol)
    permutation = list(range(nr_locations))
    random.shuffle(permutation)
    individual = [0] * nr_locations
    truck = 0
    for p in permutation:
        individual[p] = truck
        truck = (truck + 1) % nr_trucks
    return individual


def createIndividual(initialsol=[]):
    cumweights = [3, 3, 3]
    funcs = [createCopy, createRandom, createUniform]
    f = random.choices(funcs, cum_weights=cumweights, k=1)[0]
    return f(initialsol)


toolbox = base.Toolbox()
# ------------------------------------------------------------------------------------------------


def init(individual):

    # from scoop import futures
    # toolbox.register("map", futures.map)

    toolbox.register("startSolution", createIndividual, individual)

    toolbox.register("individualCreator", tools.initIterate,
                     creator.Individual, toolbox.startSolution)

    toolbox.register("populationCreator", tools.initRepeat,
                     list, toolbox.individualCreator)

    toolbox.register("evaluate", evaluateCandidate)

    toolbox.register("select", tools.selTournament,
                     tournsize=eam.TOURNAMENT_SIZE)

    toolbox.register("mate", tools.cxTwoPoint)

    toolbox.register("mutate", tools.mutUniformInt, low=0,
                     up=max(individual), indpb=1/len(individual))

    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=s.number/len(s.locations))
# ------------------------------------------------------------------------------------------------


def executeEA():
    population = toolbox.populationCreator(n=eam.POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(eam.HALL_OF_FAME_SIZE)

    # Unregister unpicklable methods before sending the toolbox.
    toolbox.unregister("startSolution")
    toolbox.unregister("individualCreator")
    toolbox.unregister("populationCreator")
    # elitism.eaSimpleWithElitism
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox,
                                                      #  mu=HALL_OF_FAME_SIZE,
                                                      #  lambda_=HALL_OF_FAME_SIZE * 4,
                                                      cxpb=eam.P_CROSSOVER,
                                                      mutpb=eam.P_MUTATION,
                                                      ngen=eam.MAX_GENERATIONS, stats=stats,
                                                      halloffame=hof, verbose=True)

    return hof, logbook, population
# ----------------------------------------------------------------------------


def plotStats(logbook: tools.Logbook):
    # plot statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')


def plotIterations(scores):
    # plot statistics:
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(scores, color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
# ------------------------------------------------------------------------------------------------


def displaySolution(trucks, logbook=None):
    if logbook != None:
        plotStats(logbook)

    for i, t in enumerate(trucks):
        trucks[i] = sat.sequence(s, t)

    s.plotData(trucks, label=True)
    print(trucks)

# ------------------------------------------------------------------------------------------------


def evolve(individual):
    init(individual)
    hof, logbook, pop = executeEA()
    return hof.items[0], logbook, pop
# ------------------------------------------------------------------------------------------------


def checkStopCondition():
    return os.path.exists("stop.txt")


def runMeta(subtrucks):
    c = s.getChromosomeFromSolution(subtrucks)
    c, logbook, _ = evolve(c)
    _ = logbook

    # --- update solution before next iteration
    subtrucks = s.getSolutionFromChromosome(c)
    return subtrucks


def main():
    c = s.findClusters()
    solution = list(s.getSolutionFromChromosome(c))

    for j in range(eam.META_ITERATIONS):

        eam.initForIteration(j)

        print(
            f"=== iteration {j} starts with {eam.META_SIZE} neighbours",
            f"and {eam.MAX_GENERATIONS} generations ===")

        # --- choose trucks and save housekeeping stuff
        anchor = s.findAnchor(trucks=solution)

        indices = s.findNeighbours(
            truck=solution[anchor], trucks=solution, D=eam.NBR_DIST, N=eam.META_SIZE)
        subtrucks = [solution[i] for i in indices]

        # --- now begins the meta ---
        subtrucks = runMeta(subtrucks)

        # --- update solution before next iteration
        for i, t in zip(indices, subtrucks):
            solution[i] = t

        if checkStopCondition():
            displaySolution(solution)

    displaySolution(solution)


# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
