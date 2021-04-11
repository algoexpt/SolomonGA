import eameta
from solomon import Solomon
import eameta
import sat
import time

PROBLEM = [("C101", 10), ("C201", 3), ("R101", 19), ("R201", 4)]
PN = PROBLEM[1]
s = Solomon(PN[0], number=PN[1])


def getChromosomeFromSolution(mytrucks):
    c = [0] * len(s.locations) - 1  # include 0.. because.
    for i, t in enumerate(mytrucks):
        for ti in t:
            c[ti] = i
    c.__delitem__(0)
    return c


def findclusters():
    indiv = s.findClusters()
    mytrucks = s.getSolutionFromCandidate(indiv)
    return seq(mytrucks)


def seq(mytrucks):
    tick = time.perf_counter()
    # 2. run scip to find most optimal sequence
    for i, t in enumerate(mytrucks):
        if t[-1] == 0:
            t.pop()
        mytrucks[i] = sat.sequence(s, t)

    tock = time.perf_counter()
    print(f"My trucks, (after {(tock-tick):.4f}s):")
    print(mytrucks)
    return mytrucks


def optimal():
    import getsol
    return getsol.getOptimalSol(PROBLEM_NAME)
    """
    [[2, 21, 73, 41, 56, 4], 
    [5, 83, 61, 85, 37, 93], 
    [14, 44, 38, 43, 13], 
    [27, 69, 76, 79, 3, 54, 24, 80], 
    [28, 12, 40, 53, 26], 
    [30, 51, 9, 66, 1], 
    [31, 88, 7, 10], 
    [33, 29, 78, 34, 35, 77],
     [36, 47, 19, 8, 46, 17], 
     [39, 23, 67, 55, 25], 
     [45, 82, 18, 84, 60, 89], 
     [52, 6], [59, 99, 94, 96],
     [62, 11, 90, 20, 32, 70], 
     [63, 64, 49, 48], 
     [65, 71, 81, 50, 68], 
     [72, 75, 22, 74, 58], 
     [92, 42, 15, 87, 57, 97], 
    [95, 98, 16, 86, 91, 100]]
    """


def onex1test():
    return[
        [0, 22, 93, 75, 2, 1, 99, 100, 97, 92, 94, 95, 98, 7, 3, 4, 89, 91, 88,
            84, 86, 83, 82, 85, 76, 71, 70, 73, 80, 79, 81, 78, 77, 96, 87, 0],
        [0, 5, 20, 24, 27, 30, 29, 6, 32, 33, 31, 35, 37, 38, 39, 36, 34, 28,
            26, 23, 18, 19, 16, 14, 12, 15, 17, 13, 25, 9, 11, 10, 8, 21, 90, 0],
        [0, 67, 63, 62, 74, 72, 61, 64, 66, 69, 68, 65, 49, 55, 54, 53, 56,
            58, 60, 59, 57, 40, 44, 46, 45, 51, 50, 52, 47, 43, 42, 41, 48, 0]
    ]


# mytrucks = optimal()
# c = getChromosomeFromSolution(mytrucks)
# mytrucks = ea.getSolutionFromCandidate(c)

mytrucks = onex1test()
# mytrucks = list(mytrucks)
print(mytrucks)
# print(s.calcScore(mytrucks))

# mytrucks = eameta.runMeta(mytrucks)
chromosome = s.getChromosomeFromSolution(mytrucks)
# print(eameta.evaluateCandidate(chromosome))

trucks = s.getSolutionFromChromosome(chromosome)
print("re-evaluated trucks", trucks)
total_score = 0
for t in trucks:
    t.sort()
    tkey = tuple(t)
    if tkey not in s.sequenced:
        seqt = sat.sequence(s, t)
        s.sequenced[tkey] = s.calcTruckScore(seqt)

    total_score += s.sequenced[tkey]

print(total_score)

# s.plotData(trucks, label=True)
