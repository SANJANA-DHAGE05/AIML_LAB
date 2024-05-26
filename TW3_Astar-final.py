#TERMWORK-3 (A*)

import heapq


def astar(graph, start, goal):
    # priority queue to store nodes to be explored
    open_list = [(0, start)]
    # dictionary to store parent nodes
    parents = {}
    # dictionary to store g values (cost from start node to current node)
    g_values = {node: float('inf') for node in graph}
    g_values[start] = 0
    # dictionary to store f values (estimated total cost from start to goal)
    f_values = {node: float('inf') for node in graph}
    f_values[start] = graph[start][1]

    iteration = 0

    while open_list:
        # get node with minimum f value
        current_f, current_node = heapq.heappop(open_list)

        # check if current node is the goal
        if current_node == goal:
            path = []
            while current_node in parents:
                path.append(current_node)
                current_node = parents[current_node]
            path.append(start)
            final_cost = g_values[goal]
            print(f"\nFinal Cost: {final_cost}")
            return path[::-1]

        # explore neighbors
        for child, cost in graph[current_node][0].items():
            # calculate tentative g value
            tentative_g = g_values[current_node] + cost
            if tentative_g < g_values[child]:
                # update parent and g values
                parents[child] = current_node
                g_values[child] = tentative_g
                f_values[child] = tentative_g + graph[child][1]
                # add child to open list
                heapq.heappush(open_list, (f_values[child], child))

        iteration += 1
        print(f"\nIteration {iteration}:")
        print("Current Path:", reconstruct_path(parents, start, current_node))
        print(f"Evaluation Function Value for {current_node}: {f_values[current_node]}")


# Function to reconstruct the path from start to goal using parent nodes
def reconstruct_path(parents, start, goal):
    path = [goal]
    while goal != start:
        goal = parents[goal]
        path.append(goal)
    return path[::-1]


# Example usage:
start_node = 'S'
goal_node = 'G'
graph = {
    'S': [{'B': 4, 'C': 3}, 14],
    'B': [{'E': 12, 'F': 5}, 12],
    'C': [{'E': 10,'D':7}, 11],
    'D': [{'E': 2}, 6],
    'E': [{'G': 5}, 4],
    'F': [{'G': 16}, 11],
    'G': [{}, 0]
}



print("\nA* Search Path:")
path = astar(graph, start_node, goal_node)
print("Final Path:", path)


'''
-------OUTPUT---------
A* Search Path:

Iteration 1:
Current Path: ['S']
Evaluation Function Value for S: 14

Iteration 2:
Current Path: ['S', 'C']
Evaluation Function Value for C: 14

Iteration 3:
Current Path: ['S', 'B']
Evaluation Function Value for B: 16

Iteration 4:
Current Path: ['S', 'C', 'D']
Evaluation Function Value for D: 16

Iteration 5:
Current Path: ['S', 'C', 'D', 'E']
Evaluation Function Value for E: 16

Iteration 6:
Current Path: ['S', 'C', 'D', 'E']
Evaluation Function Value for E: 16

Final Cost: 17
'''
