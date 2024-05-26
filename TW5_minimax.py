#TERMWORK-5 (MINIMAX)

import math

def minimax(curDepth, nodeIndex, maxTurn, scores, targetDepth):
    # Base case: targetDepth reached
    if curDepth == targetDepth:
        return scores[nodeIndex]

    if maxTurn:
        left_child = nodeIndex * 2
        right_child = nodeIndex * 2 + 1
        left_score = minimax(curDepth + 1, left_child, False, scores, targetDepth)
        right_score = minimax(curDepth + 1, right_child, False, scores, targetDepth)
        print(f"Node {nodeIndex}: Max({left_score}, {right_score})")
        return max(left_score, right_score)
    else:
        left_child = nodeIndex * 2
        right_child = nodeIndex * 2 + 1
        left_score = minimax(curDepth + 1, left_child, True, scores, targetDepth)
        right_score = minimax(curDepth + 1, right_child, True, scores, targetDepth)
        print(f"Node {nodeIndex}: Min({left_score}, {right_score})")
        return min(left_score, right_score)

# Driver code
scores = [-1, 4, 2, 6, -3, -5, 0, 7]
treeDepth = int(math.log(len(scores), 2))

optimal_value = minimax(0, 0, True, scores, treeDepth)
print(f"The optimal value is: {optimal_value}")


'''
---------OUTPUT----------
Node 0: Max(-1, 4)
Node 1: Max(2, 6)
Node 0: Min(4, 6)
Node 2: Max(-3, -5)
Node 3: Max(0, 7)
Node 1: Min(-3, 7)
Node 0: Max(4, -3)
The optimal value is: 4
'''
