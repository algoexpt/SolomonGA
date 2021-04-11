# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 03:55:56 2021

@author: Ajay
"""

from ortools.sat.python import cp_model
import solomon
import math


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        # print("value:", self.ObjectiveValue())
        # for v in self.__variables:
        #     if self.Value(v) > 0:
        #         print('%s=%i' % (v, self.Value(v)), end=' ')
        # print()

    def solution_count(self):
        return self.__solution_count


def sequence(s: solomon.Solomon, truck=[]):
    # Creates the model.
    model = cp_model.CpModel()

    time = {}
    late = {}
    edge = {}
    seq = {}
    dist = {}

    slack = 4

    if truck[0] != 0:
        truck.insert(0, 0)

    if truck[-1] == 0:
        _ = truck.pop()

    # print("====== TRUCK: ==", truck)

    maxdist = math.ceil(sum([max(s.distances[m]) for m in truck]))

    # Creates the variables.
    for n in truck:
        starttime = math.floor(s.locations[n]['start'])
        endtime = math.ceil(s.locations[n]['end'])

        # time var
        mindist = math.ceil(s.distances[0][n])
        time[n] = model.NewIntVar(
            max(starttime, mindist), slack*endtime, name=f'time-{n}')
        late[n] = model.NewIntVar(0, (slack-1)*endtime, f'late-{n}')
        dist[n] = model.NewIntVar(mindist, maxdist, f"dist-{n}")

        # also set up lateness for penalty
        model.Add(late[n] >= time[n] - endtime)

        # sequencevar
        if n > 0:
            seq[n] = model.NewIntVar(1, len(truck) - 1, f'seq-{n}')
        else:
            seq[n] = model.NewIntVar(len(truck), len(truck), f'seq-{n}')

    for m in truck:
        if m == 0:
            # no need to consider starts from 0, we already set the right lowerbounds
            continue
        for n in truck:
            if n == m:
                continue
            edge[(m, n)] = model.NewBoolVar(f'edge({m}-{n})')

    for (m, n) in edge:
        if m == 0:
            # time and dist already have a lowerbound of travel distance from 0
            # nothing to do here.
            continue

        e = edge[(m, n)]

        delta = math.ceil(s.distances[m][n] + s.locations[m]['service'])
        model.Add(time[n] - time[m] >= delta).OnlyEnforceIf(e)
        model.Add(dist[n] - dist[m] >=
                  math.ceil(s.distances[m][n])).OnlyEnforceIf(e)

        # Tie the edge to the sequence
        model.Add(seq[n] < seq[m]).OnlyEnforceIf(e.Not())
        model.Add(seq[m] < seq[n]).OnlyEnforceIf(e)

    # sequence number is different for all of them
    model.AddAllDifferent(list(seq.values()))

    for m in truck:
        incoming = [edge[(n, m)]
                    for n in truck[1:] if n != m]  # no need for source
        model.Add(sum(incoming) == seq[m] - 1)

        if m > 0:
            outgoing = [edge[(m, n)] for n in truck if n != m]
            model.Add(sum(outgoing) == len(truck) - seq[m])

    model.Minimize(dist[0] + sum([s.LATE_PENALTY * late[l] for l in late]))

    # model.ExportToFile("problem.txt")
    # Create a solver and solve.
    solution_printer = VarArraySolutionPrinter(list(edge.values()))
    status, solver = Solve(model, solution_printer, 5)
    # status = solver.SearchForAllSolutions(model, solution_printer)

    statusname = solver.StatusName(status)

    # print('Status = %s' % statusname)
    # print('Number of solutions found: %i' % solution_printer.solution_count())

    if statusname in ['FEASIBLE', 'OPTIMAL']:
        truck.sort(key=lambda x: solver.Value(seq[x]))
        truck.insert(0, 0)
        # print('result route = ', truck)
        # print("time at 0:", solver.Value(time[0]))
    else:
        # print(f"solver status:{statusname}")
        # print("truck:", truck)
        None

    return truck


def Solve(model, solution_printer, timeinsec):

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeinsec
    solver.parameters.num_search_workers = 16
    #solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    solver.parameters.search_branching = cp_model.FIXED_SEARCH

    # print("Starting solve...")
    # status = solver.SolveWithSolutionCallback(model, solution_printer)
    status = solver.Solve(model)
    return status, solver
