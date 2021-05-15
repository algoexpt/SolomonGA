# SOLOMON BENCHMARKS using DEAP and ORT

I've written a blog post about this project, worth having a read if you're interested: https://www.notion.so/ajayarn/Using-Evolutionary-Algorithms-for-the-CVRP-0bfa73e06d5247a1bfa4988d33ef304b

# High level design
- GA (DEAP) as a meta-heuristic, CP-Sat solver as a sub optimizer
- GA generates chromosomes where each node is a city, the number of each node is the truck that the node is on
- CP-SAT takes the subset of nodes per truck and sequences them
- Use K-Means clustering for generating initial population

# Results
- Within 5% for R101, C101. Within 10% for RC.
