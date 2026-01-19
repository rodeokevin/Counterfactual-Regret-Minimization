import numpy as np
import random

# 3.5 Exercise: 1-Die-Versus-1-Die Dudo

# Dudo Trainer
class DudoTrainer:
    # Dudo definitions
    def __init__(self):
        self.NUM_SIDES = 6
        self.NUM_ACTIONS = (2*self.NUM_SIDES) + 1
        self.DUDO = self.NUM_ACTIONS - 1
        self.claimNum = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        self.claimRank = np.array([2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1])
        self.nodeMap = {}
    
    # Convert Dudo claim history to a string
    def claimHistoryToString(self, isClaimed):
        s = ""
        for a in range(self.NUM_ACTIONS):
            if (a == self.DUDO and isClaimed[a]):
                s += ',DUDO'
                continue
            if (isClaimed[a]):
                if (len(s) > 0):
                    s += ','
                s += str(self.claimNum[a])
                s += '*'
                s += str(self.claimRank[a])
        return s

    # Convert Dudo information set to an integer
    def infoSetToInteger(self, playerRoll, isClaimed):
        infoSetNum = playerRoll
        for a in range(self.NUM_ACTIONS-1, -1, -1):
            infoSetNum = (infoSetNum << 1) + (1 if isClaimed[a] else 0)
        return infoSetNum
    
    # Information set node class definition
    class Node:
        # minI, maxI inclusive
        def __init__(self, minI, maxI):
            self.NUM_ACTIONS = (maxI - minI) + 1
            self.MIN_ACTION = minI
            self.MAX_ACTION = maxI
            self.infoSet = ""
            self.regretSum = np.zeros(self.NUM_ACTIONS)
            self.strategy = np.zeros(self.NUM_ACTIONS)
            self.strategySum = np.zeros(self.NUM_ACTIONS)
        
        def getStrategy(self, realizationWeight):
            self.strategy = np.clip(self.regretSum, a_min=0, a_max=None)
            normalizingSum = sum(self.strategy)
            if (normalizingSum > 0):
                self.strategy /= normalizingSum
            else:
                self.strategy = np.repeat(1.0 / self.NUM_ACTIONS, self.NUM_ACTIONS)
            self.strategySum += realizationWeight * self.strategy
            return self.strategy

        def getAverageStrategy(self):
            normalizingSum = sum(self.strategySum)
            if (normalizingSum > 0):
                return self.strategySum / normalizingSum
            else:
                return np.repeat(1.0 / self.NUM_ACTIONS, self.NUM_ACTIONS)
        
        def toString(self):
            strat = np.array2string(
                self.getAverageStrategy(),
                precision=2,
                suppress_small=True,
                floatmode="fixed"
            )
            return f"{self.infoSet:>4}: {strat}"
    
    def train(self, iterations):
        util = 0.0
        for i in range(iterations):
            if (i % 1000 == 0):
                print("Iteration: ", i)
            roll1 = random.randint(1,6)
            roll2 = random.randint(1,6)
            util += self.cfr([roll1, roll2],[False] * self.NUM_ACTIONS, 1, 1, -1)
        print("Average game value: " + str(util / iterations))
        with open('output.txt', 'w') as f:
            for key in sorted(self.nodeMap):
                print(self.nodeMap[key].toString(), file=f)
    
    # Counterfactual regret minimization iteration
    # history is a boolean array of length self.NUM_ACTIONS
    def cfr(self, nums, history, p0, p1, lastAction):
        plays = sum(history)
        player = plays % 2
        # Return payoff for terminal states, i.e. when DUDO is called
        if (plays > 1 and lastAction == self.DUDO):
            # Get last claim before DUDO
            lastClaim = -1
            for a in range(self.DUDO - 1, -1, -1):
                if history[a]:
                    lastClaim = a
                    break
            count = self.claimNum[lastClaim]
            rank = self.claimRank[lastClaim]
            for n in nums:
                if (n == 1 or n == rank):
                    count -= 1
            # Rank count was exact or more than the claimed count
            if (count <= 0):
                return 1
            # Rank count is less than the claimed count
            elif (count > 0):
                return -1
        infoSetStr = str(nums[player]) + self.claimHistoryToString(history)
        infoSetNum = self.infoSetToInteger(nums[player], history)
        # Get information set node or create it if nonexistant
        if (infoSetNum in self.nodeMap):
            node = self.nodeMap[infoSetNum]
        else:
            # The valid actions are the indices from lastAction+1 onwards, and don't allow DUDO action until at least 1 action has been played
            node = self.Node(lastAction+1, (self.DUDO if plays > 0 else self.DUDO - 1))
            node.infoSet = infoSetStr
            self.nodeMap[infoSetNum] = node
        # For each action, recursively call cfr with additional history and probability
        strategy = node.getStrategy(p0 if player == 0 else p1)
        util = np.zeros(node.NUM_ACTIONS)
        nodeUtil = 0
        for a in range(node.NUM_ACTIONS):
            nextHistory = history.copy()
            nextHistory[node.MIN_ACTION + a] = True
            if (player == 0):
                util[a] = -self.cfr(nums, nextHistory, p0 * strategy[a], p1, node.MIN_ACTION + a)
            else:
                util[a] = -self.cfr(nums, nextHistory, p0, p1 * strategy[a], node.MIN_ACTION + a)
            # nodeUtil += strategy[a] * util[a]
        nodeUtil = np.sum(strategy * util)
        # For each action, compute and accumulate counterfactual regret
        node.regretSum += (p1 if player == 0 else p0) * (util - nodeUtil)
        return nodeUtil

# test = DudoTrainer()
# arr = np.array([True, False, True, False, True, False, False, False, True, True, False, False, True])
# print(test.claimHistoryToString(arr))
# print(test.infoSetToInteger(0, arr))

trainer = DudoTrainer()
trainer.train(10_000)