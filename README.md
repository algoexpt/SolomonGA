<<<<<<< HEAD
# SOLOMON BENCHMARKS using DEAP and ORT

This is a placeholder README, I will update it with more details at a later point.

# High level design
- GA (DEAP) as a meta-heuristic, CP-Sat solver as a sub optimizer
- GA generates chromosomes where each node is a city, the number of each node is the truck that the node is on
- CP-SAT takes the subset of nodes per truck and sequences them
- Use K-Means clustering for generating initial population

# Results
- Within 5% for R101, C101. Within 10% for RC.
