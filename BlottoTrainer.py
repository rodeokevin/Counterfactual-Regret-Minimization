import numpy as np
from numpy.random import choice

# 2.6 Exercise: Colonel Blotto

# Helper function to generate all possible actions
def compositions(n,k):
  if n < 0 or k < 0:
    return
  elif k == 0:
    # the empty sum, by convention, is zero, so only return something if n is zero
    if n == 0:
      yield []
    return
  elif k == 1:
    yield [n]
    return
  else:
    for i in range(0,n+1):
      for comp in compositions(n-i,k-1):
        yield [i] + comp

class Player:
    def __init__(self, numActions):
        self.regretSum = np.zeros(numActions)
        self.strategy = np.zeros(numActions)
        self.strategySum = np.zeros(numActions)

class BattleTrainer:
    def __init__(self):
        self.ALLOCATIONS = np.array([ c for c in compositions(5,3) ])
        self.NUM_ACTIONS = len(self.ALLOCATIONS)
        self.possible_actions = np.arange(self.NUM_ACTIONS)
        self.p0 = Player(self.NUM_ACTIONS)
        self.p1 = Player(self.NUM_ACTIONS)
        # self.fixedStrategy = np.append(np.array([1.0]), np.zeros(20)) 
        # self.fixedStrategy = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    # Get current mixed strategy through regret-matching
    def getStrategy(self, regretSum):
        strategy = np.clip(regretSum, a_min=0, a_max=None)
        normalizingSum = sum(strategy)
        if (normalizingSum > 0):
            strategy /= normalizingSum
        else:
            strategy = np.repeat(1.0 / self.NUM_ACTIONS, self.NUM_ACTIONS)
        return strategy

    # Get the next action based on a random number and the cumulative probabilities
    def getAction(self, strategy):
        return choice(self.possible_actions, p=strategy)
    
    # Compute utility for all possible actions given the other player's action
    def computeUtility(self, oppAction):
        oppAlloc = self.ALLOCATIONS[oppAction]
    
        larger  = np.sum(self.ALLOCATIONS > oppAlloc, axis=1)
        smaller = np.sum(self.ALLOCATIONS < oppAlloc, axis=1)

        utilities = np.sign(larger - smaller)
        return utilities

    # Training algorithm
    def train(self, iterations):
        actionUtility0 = np.zeros(self.NUM_ACTIONS)
        actionUtility1 = np.zeros(self.NUM_ACTIONS)
        for _ in range(iterations):
            # Get the regret-matched strategies for each player
            strategy0 = self.getStrategy(self.p0.regretSum)
            strategy1 = self.getStrategy(self.p1.regretSum)
            self.p0.strategySum += strategy0
            self.p1.strategySum += strategy1
            action0 = self.getAction(strategy0)
            action1 = self.getAction(strategy1)

            # Compute the utilities of each possible action
            actionUtility0 = self.computeUtility(action1)
            actionUtility1 = self.computeUtility(action0)
            
            # Accumulate the regrets
            self.p0.regretSum += actionUtility0 - actionUtility0[action0]
            self.p1.regretSum += actionUtility1 - actionUtility1[action1]

    # Compute the average strategy accross all iterations
    def getAverageStrategy(self, strategySum):
        normalizingSum = sum(strategySum)
        if (normalizingSum > 0):
            return strategySum / normalizingSum
        else:
            return np.repeat(1.0 / self.NUM_ACTIONS, self.NUM_ACTIONS)

trainer = BattleTrainer()
trainer.train(1000000)
p0ms = trainer.getAverageStrategy(trainer.p0.strategySum)
# print(p0ms)
for i,s in enumerate(p0ms):
    print(trainer.ALLOCATIONS[i], p0ms[i])
p1ms = trainer.getAverageStrategy(trainer.p1.strategySum)
print("--------------------------------------")
for i,s in enumerate(p1ms):
    print(trainer.ALLOCATIONS[i], p1ms[i])
