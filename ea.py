# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:25:46 2021

@author: Ajay
"""

from solomon import Solomon
from deap import base, tools, creator, algorithms
import random
import array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sat
import elitism
from scoop import futures
import math

PROBLEM_NAME = "R101"
s = Solomon(PROBLEM_NAME, number=19)

random.seed(64)

if "FitnessMin" not in dir(creator):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if "Individual" not in dir(creator):
    creator.create("Individual", array.array, typecode='i',
                fitness=creator.FitnessMin)
                
# ---------------------------------------------------------------------------
TOURNAMENT_SIZE = 2
POPULATION_SIZE = 300
P_CROSSOVER = 0.8
P_MUTATION = 0.2
MAX_GENERATIONS = 500
HALL_OF_FAME_SIZE = 30
# ---------------------------------------------------------------------------
# algo = elitism.eaSimpleWithElitism
# algo = algorithms.eaSimple
# ---------------------------------------------------------------------------

   
def getSolutionFromCandidate(candidate:list):
    trucks = []
    for t in range(s.number):
        locs = [i + 1 for i in range(len(candidate)) if candidate[i] == t]
        if len(locs) > 0:
            locs.insert(0,0)
            trucks.append(locs)
    return trucks

def evaluateCandidate(candidate: list):
    # 1. collect locations per vehicle
    trucks = getSolutionFromCandidate(candidate)

    # 2. run scip to find most optimal sequence
    for i, t in enumerate(trucks):
        trucks[i] = sat.sequence(s, t)

    # 3. actually evaluate sequence calcScore
    return s.calcScore(trucks),

def createRandom(nr_trucks, nr_locations):
        choice = random.choices(population=range(nr_trucks), k=nr_locations)
        return choice

def createUniform(nr_trucks, nr_locations):
    permutation = list(range(nr_locations))
    random.shuffle(permutation)
    individual = [0 for _ in range(nr_locations)]
    truck = 0
    for p in permutation:
        individual[p] = truck
        truck = ( truck + 1 ) % nr_trucks  
    return individual  

def createSequence(nr_trucks, nr_locations):
    share = math.floor(nr_locations / nr_trucks)
    indiv = []
    truck = 0
    c = 0
    for i in range(nr_locations):
        if truck + 1 < nr_trucks and c >= share:
            c = 0
            truck = (truck + 1)%nr_trucks
        indiv.append(truck)
        c += 1
    return indiv

def createClusters(nr_trucks, nr_locations):
    indiv = s.findClusters()
    return indiv

def createIndividual(nr_trucks, nr_locations):
    create_weights = [1, 1, 1, 1]
    create_funcs = [createSequence, createClusters, createUniform, createRandom]
    f = random.choices(create_funcs, weights=create_weights)[0]
    return f(nr_trucks, nr_locations)


toolbox = base.Toolbox()


def init():
  
    toolbox.register("map", futures.map)

    toolbox.register("randomOrder", createIndividual, s.number, len(s.locations) - 1)  
    toolbox.register("individualCreator", tools.initIterate,
                        creator.Individual, toolbox.randomOrder)
    toolbox.register("populationCreator", tools.initRepeat,
                        list, toolbox.individualCreator)

    toolbox.register("evaluate", evaluateCandidate)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=s.number/len(s.locations))
    toolbox.register("mutate", tools.mutUniformInt, low=0,
                        up=s.number-1, indpb=s.number/len(s.locations))
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=s.number/len(s.locations))
# ------------------------------------------------------------------------------------------------

def executeEA():
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Unregister unpicklable methods before sending the toolbox.
    toolbox.unregister("randomOrder")
    toolbox.unregister("individualCreator")
    toolbox.unregister("populationCreator")
    # elitism.eaSimpleWithElitism
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, 
                               cxpb=P_CROSSOVER,
                               mutpb=P_MUTATION,
                               ngen=MAX_GENERATIONS, stats=stats,
                               halloffame=hof, verbose=True)

    return hof, logbook, population
# ----------------------------------------------------------------------------
def plotStats(logbook):
    # plot statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
# ------------------------------------------------------------------------------------------------


def displaySolution(hof, logbook, population):
    # print hall of fame members info:
    # print("- Best solutions are:")
    # print(hof.items)
    # for i, sol in enumerate hof.items:
    #     # , " -> ", hof.items[i])
    #     print(i, ": ", sol.fitness.values[0])

    plotStats(logbook)
    trucks = getSolutionFromCandidate(hof.items[0])
    for i, t in enumerate(trucks):
        trucks[i] = sat.sequence(s, t)
    s.plotData(trucks)
    print(hof.items[0])

    # plt.show()
# ------------------------------------------------------------------------------------------------


def evolve():
    init()
    hof, logbook, pop = executeEA()
    displaySolution(hof, logbook, pop)

if __name__ == "__main__":
    evolve()