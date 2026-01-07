import random

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
        self.regretSum = [0.0] * numActions
        self.strategy = [0.0] * numActions
        self.strategySum = [0.0] * numActions

class BattleTrainer:
    # Definitions
    ALLOCATIONS = [ c for c in compositions(5,3) ]
    NUM_ACTIONS = len(ALLOCATIONS)

    def __init__(self):
        self.p0 = Player(self.NUM_ACTIONS)
        self.p1 = Player(self.NUM_ACTIONS)

    # Get current mixed strategy through regret-matching
    def getStrategy(self, player):
        normalizingSum = 0.0
        for i in range(self.NUM_ACTIONS):
            player.strategy[i] = max(player.regretSum[i], 0)
            normalizingSum += player.strategy[i]
        for i in range(self.NUM_ACTIONS):
            if (normalizingSum > 0):
                player.strategy[i] /= normalizingSum
            else:
                player.strategy[i] = 1.0 / self.NUM_ACTIONS
            player.strategySum[i] += player.strategy[i]
        return player.strategy

    # Get the next action based on a random number and the cumulative probabilities
    def getAction(self, strategy):
        r = random.random()
        i = 0
        cumulativeProb = 0.0
        while (i < self.NUM_ACTIONS - 1):
            cumulativeProb += strategy[i]
            if (r < cumulativeProb):
                break
            i += 1
        return i
    
    # Compute utility for a specific action in the point of view of p0action
    def computeUtility(self, p0action, p1action):
        if (p0action == p1action):
            return 0
        # First get how each player has decided to separate their troops
        pstrategy0 = self.ALLOCATIONS[p0action] 
        pstrategy1 = self.ALLOCATIONS[p1action]
        points0 = 0
        points1 = 0
        for i in range(len(pstrategy0)):
            if (pstrategy0[i] > pstrategy1[i]):
                points0 += 1
            elif (pstrategy0[i] < pstrategy1[i]):
                points1 += 1
        if (points0 == points1):
            return 0
        elif (points0 > points1):
            return 1
        else:
            return -1


    # Training algorithm
    def train(self, iterations):
        actionUtility0 = [0.0] * self.NUM_ACTIONS
        actionUtility1 = [0.0] * self.NUM_ACTIONS
        for i in range(iterations):
            # Get the regret-matched strategies for each player
            strategy0 = self.getStrategy(self.p0)
            strategy1 = self.getStrategy(self.p1)
            action0 = self.getAction(strategy0)
            action1 = self.getAction(strategy1)
            # Compute the utilities of each possible action
            for battle in range(self.NUM_ACTIONS):
                actionUtility0[battle] = self.computeUtility(battle, action1)
                actionUtility1[battle] = self.computeUtility(battle, action0)
            # Accumulate the regrets
            for j in range(self.NUM_ACTIONS):
                self.p0.regretSum[j] += actionUtility0[j] - actionUtility0[action0]
                self.p1.regretSum[j] += actionUtility1[j] - actionUtility1[action1]


    # Compute the average strategy accross all iterations
    def getAverageStrategy(self, player):
        avgStrategy = [0.0] * self.NUM_ACTIONS
        normalizingSum = 0.0
        for i in range(self.NUM_ACTIONS):
            normalizingSum += player.strategySum[i]
        for i in range(self.NUM_ACTIONS):
            if (normalizingSum > 0):
                avgStrategy[i] = player.strategySum[i] / normalizingSum
            else:
                avgStrategy[i] = 1.0 / self.NUM_ACTIONS
        return avgStrategy

trainer = BattleTrainer()
trainer.train(1000000)
p0ms = trainer.getAverageStrategy(trainer.p0)
# print(p0ms)
for i,s in enumerate(p0ms):
    print(trainer.ALLOCATIONS[i], p0ms[i])
p1ms = trainer.getAverageStrategy(trainer.p1)
print("--------------------------------------")
for i,s in enumerate(p1ms):
    print(trainer.ALLOCATIONS[i], p1ms[i])
