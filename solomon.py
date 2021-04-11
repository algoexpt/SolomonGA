# from data.text2json2 import BASE_DIR
import numpy as np
import json
import io
import matplotlib.pyplot as plt
import random


class Solomon:
    """This class encapsulates the VRPTW problem
    """

    def __init__(self, problemName, number=1000):
        """
        Creates an instance of a Solomon problem

        :param name: name of the problem
        """

        # initialize instance variables:
        self.name = problemName
        self.locations = []
        self.distances = []
        self.number = number
        self.capacity = 0
        self.LATE_PENALTY = 100  # per unit
        self.LOAD_PENALTY = 1000  # per unit
        self.sequenced = {}
        self.metaChromosome = []

        # initialize the data:
        self.__initData()

    def __len__(self):
        """
        returns the length of the underlying TSP
        :return: the length of the underlying TSP (number of cities)
        """
        return len(self.locations)

    def __initData(self):
        DATA_DIR = "./data/json/"
        # attempt to read json data:
        obj = None
        with io.open(DATA_DIR + self.name + ".json", 'rt', newline='') as file_object:
            obj = json.load(file_object)

        if obj != None and self.name == obj['name']:
            self.locations = obj['locs']
            self.distances = obj['dist']
            self.number = min(obj['number'], self.number)
            self.capacity = obj['capacity']
            # print(obj["name"], locs[0], locs[99], dist[0][99])

    def plotData(self, solution=[], centroids=[], label=False):
        """plots the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: the resulting plot
        """
        plt.figure(0)

        np.random.seed(42)
        x = [l['x'] for l in self.locations]
        y = [l['y'] for l in self.locations]
        colors = np.random.rand(len(self.locations))
        # colors[0] = 10.0
        area = [10 + (l['demand'] * 2) for l in self.locations]

        plt.scatter(x, y, s=area, c=colors, alpha=0.7)
        plt.scatter([x[0]], [y[0]], marker="s")  # show the depot as a square
        if label:
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.annotate(str(i), (xi, yi))

        plt.title(self.calcScore(solution))

        for truck in solution:
            if truck[0] != 0:
                truck.insert(0, 0)
            if truck[-1] != 0:
                truck.append(0)

            xs = [self.locations[i]['x'] for i in truck]
            ys = [self.locations[i]['y'] for i in truck]

            # plot a line between each pair of consecutive cities:
            plt.plot(xs, ys)

        if len(centroids) > 0:
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker="+")  # the centroid as a +

        plt.show()

        return plt

    def calcTruckScore(self, truck):
        p = 0
        time = 0
        load = 0

        total_penalty = 0
        total_dist = 0

        for q in truck + [0]:
            dist = self.distances[p][q]
            total_dist += dist
            time += dist

            load += self.locations[q]['demand']

            late = time - self.locations[q]['end']
            if late > 0:
                # print(f"late by {late} between {p} and {q}")
                total_penalty += self.LATE_PENALTY * late

            if time < self.locations[q]['start']:
                time = self.locations[q]['start']

            time = time + self.locations[q]['service']
            p = q

            # print("time:", q, time)

        if load > self.capacity:
            total_penalty += self.LOAD_PENALTY * (load - self.capacity)

        return total_dist + total_penalty

    def calcScore(self, trucks: list, penalize=False):
        return sum([self.calcTruckScore(t) for t in trucks])

    def findClusters(self):
        from sklearn.cluster import KMeans
        # import numpy as np
        points = [[float(l['x']), float(l['y'])] for l in self.locations[1:]]
        kmeans = KMeans(n_clusters=self.number, random_state=0).fit(points)
        labels = kmeans.labels_
        # clusters = kmeans.cluster_centers_
        # self.plotData(centroids=clusters)
        return labels

    def findNeighbours(self, truck, trucks, D, N):

        outerpoints = set(range(len(self.locations))) - set(truck)
        # [p for p in range(len(self.locations)) if p not in truck]

        neighbours = {}

        for c in truck:
            if c == 0:
                continue

            cx, cy = self.locations[c]['x'], self.locations[c]['y']

            # fast distance check:
            for d in outerpoints:
                if d == 0:
                    continue

                dx, dy = self.locations[d]['x'], self.locations[d]['y']

                if np.absolute(dx - cx) <= D or np.absolute(dy - cy) <= D:
                    nbr = 0

                    for j, t in enumerate(trucks):
                        if d in t:
                            nbr = j
                            break

                    neighbours[nbr] = neighbours.get(nbr, 0) + 1

        # nrs = random.choices(list(neighbours.keys()), weights=list(neighbours.values()), k=N)
        nrs = []
        if len(neighbours) < N:
            nrs = list(neighbours.keys())
        else:
            nrs = random.sample(list(neighbours.keys()), k=N)
        return nrs

    def findAnchor(self, trucks: list):

        scores = [(i, self.calcTruckScore(t)) for i, t in enumerate(trucks)]
        scores.sort(key=lambda x: x[1], reverse=True)
        w = random.sample(scores[:3], k=1)
        return w[0][0]

        # ---- random anchor selection
        # worsttruck = random.sample(range(len(trucks)), k=1)
        # return worsttruck[0]

    def getChromosomeFromSolution(self, trucks):
        self.metaChromosome = [-1] * len(self.locations)
        for i, truck in enumerate(trucks):
            for l in truck:
                self.metaChromosome[l] = i

        ret = [i for i in self.metaChromosome if i > -1]
        return ret

    def getSolutionFromChromosome(self, chromosome):
        if self.metaChromosome == [] or len(chromosome) + 1 == len(self):
            self.metaChromosome = [-1] + list(chromosome)

        trucks = {}

        locations = [i for i, t in enumerate(
            self.metaChromosome) if t > -1]
        # print("getSolutionFromChromosome", locations)
        for i, t in enumerate(chromosome):
            if t not in trucks:
                trucks[t] = []
            trucks[t].append(locations[i])

        return trucks.values()
