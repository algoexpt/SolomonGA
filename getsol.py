def parseSolution(solution=[]):
    optimal_route = []
    lines = solution.split('\\r\\n')
    for line in lines:
        if line == '':
            continue
        words = line.split(":")
        if len(words) < 2:
            print("words: ", words)
            continue
        route = list(map(int, words[1].strip().split(" ")))
        print("Route", route)
        optimal_route.append(route)

    return optimal_route

def getOptimalSol(problemName:str):
    problemName = problemName.lower()
    url = f"https://www.sintef.no/contentassets/adf48e65e3a84dd6871eb7586707675d/{problemName}.txt"

    import urllib3
    response = str(urllib3.PoolManager().request("GET", url).data).lower()
    solution = response.split("solution")[1]
    return parseSolution(solution=solution)


if __name__=="__main__":
    print(getOptimalSol("R101"))