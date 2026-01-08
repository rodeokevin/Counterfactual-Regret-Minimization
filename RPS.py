import numpy as np
from numpy.random import choice

# 2.4 Worked Example: Rock-Paper-Scissors

class RPSTrainer:

    def __init__(self):
        self.NUM_ACTIONS = 3
        self.possible_actions = np.arange(self.NUM_ACTIONS)
        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.strategy = np.zeros(self.NUM_ACTIONS)
        self.strategySum = np.zeros(self.NUM_ACTIONS)
        self.oppStrategy = np.array([0.34, 0.33, 0.33]) # Arbitrary strategy for the opponent

    # Get current mixed strategy through regret-matching
    def getStrategy(self):
        normalizingSum = 0.0
        self.strategy = np.clip(self.regretSum, a_min=0, a_max=None)
        normalizingSum = sum(self.strategy)
        if (normalizingSum > 0):
            self.strategy /= normalizingSum
        else:
            self.strategy = np.repeat(1.0 / self.NUM_ACTIONS, self.NUM_ACTIONS)
        self.strategySum += self.strategy
        return self.strategy

    # Get the next action based on a random number and the cumulative probabilities
    def getAction(self, strategy):
        return choice(self.possible_actions, p=strategy)

    # Training algorithm
    def train(self, iterations):
        actionUtility = np.zeros(self.NUM_ACTIONS)
        for _ in range(iterations):
            # Get the regret-matched strategies for each player
            strategy = self.getStrategy()
            myAction = self.getAction(strategy)
            oppAction = self.getAction(self.oppStrategy)
            # Compute the utilities of each possible action
            actionUtility[oppAction] = 0
            actionUtility[(oppAction + 1) % self.NUM_ACTIONS] = 1
            actionUtility[(oppAction - 1) % self.NUM_ACTIONS] = -1
            # Accumulate the regrets
            self.regretSum += actionUtility - actionUtility[myAction]


    # Compute the average strategy accross all iterations
    def getAverageStrategy(self):
        normalizingSum = sum(self.strategySum)
        if (normalizingSum > 0):
            return self.strategySum / normalizingSum
        else:
            return np.repeat(1.0 / self.NUM_ACTIONS, self.NUM_ACTIONS)

# 2.5 Exercise: RPS Equilibrium

class Player:
    def __init__(self, numActions):
        self.regretSum = np.zeros(numActions)
        self.strategy = np.zeros(numActions)
        self.strategySum = np.zeros(numActions)

class RPSTrainer2:

    def __init__(self):
        self.NUM_ACTIONS = 3
        self.possible_actions = np.arange(self.NUM_ACTIONS)
        self.p0 = Player(self.NUM_ACTIONS)
        self.p1 = Player(self.NUM_ACTIONS)

    # Get current mixed strategy through regret-matching
    def getStrategy(self, regretSum):
        strategy = np.clip(regretSum, a_min=0, a_max=None)
        normalizingSum = np.sum(strategy)
        if (normalizingSum > 0):
            strategy /= normalizingSum
        else:
            strategy = np.repeat(1.0 / self.NUM_ACTIONS, self.NUM_ACTIONS)
        # player.strategySum[i] += player.strategy[i]
        return strategy

    # Get the next action based on a random number and the cumulative probabilities
    def getAction(self, strategy):
        return choice(self.possible_actions, p=strategy)

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
            actionUtility0[action1] = 0
            actionUtility1[action0] = 0
            actionUtility0[(action1 + 1) % self.NUM_ACTIONS] = 1
            actionUtility0[(action1 - 1) % self.NUM_ACTIONS] = -1
            actionUtility1[(action0 + 1) % self.NUM_ACTIONS] = 1
            actionUtility1[(action0 - 1) % self.NUM_ACTIONS] = -1
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
        

# trainer = RPSTrainer()
# trainer.train(1_000_000)
# print(trainer.getAverageStrategy())

trainer = RPSTrainer2()
trainer.train(1_000_000)
print(trainer.getAverageStrategy(trainer.p0.strategySum))
print(trainer.getAverageStrategy(trainer.p1.strategySum))