#TERMWORK-1 (DFID)

from collections import defaultdict
graph = defaultdict(list)

def addEdge(u, v):
    graph[u].append(v)

def dfs(start, goal, depth):
    print(start, end=" -> ")
    if start == goal:
        return True
    if depth <= 0:
        return False
    for i in graph[start]:
        if dfs(i,  goal, depth-1):
            return True
    return False


def dfid(start, goal, maxDepth):
    #print("Start node : ", start, " Goal node : ", goal)
    for i in range(maxDepth):
        print("\n\nDFID at level : ", i+1)
        print("Path : ", end=' ')
        isPathFound = dfs(start, goal, i)
    if isPathFound:
        print("\n\nGoal node found!")
        return
    else:
        print("\n\nGoal node not found!")


graph = defaultdict(list)
addEdge('A', 'B',)
addEdge('A', 'C')
addEdge('A', 'D')
addEdge('B', 'E')
addEdge('B', 'F')
addEdge('C', 'G')
addEdge('D', 'H')
addEdge('E', 'I')
addEdge('F', 'J')
addEdge('F', 'K')
addEdge('G', 'L')
addEdge('H', 'M')
addEdge('H', 'N')
addEdge('K', 'O')
addEdge('K', 'P')
addEdge('L', 'R')
addEdge('M', 'S')
addEdge('N', 'T')
addEdge('N', 'U')

dfid('A', 'T', 5)



'''
*******OUTPUT**********
DFID at level :  1
Path :  A -> 

DFID at level :  2
Path :  A -> B -> C -> D -> 

DFID at level :  3
Path :  A -> B -> E -> F -> C -> G -> D -> H -> 

DFID at level :  4
Path :  A -> B -> E -> I -> F -> J -> K -> C -> G -> L -> D -> H -> M -> N -> 

DFID at level :  5
Path :  A -> B -> E -> I -> F -> J -> K -> O -> P -> C -> G -> L -> R -> D -> H -> M -> S -> N -> T -> 

Goal node found!
'''

