import random

# 2.4 Worked Example: Rock-Paper-Scissors

class RPSTrainer:
    # Definitions
    ROCK, PAPER, SCISSORS, NUM_ACTIONS = 0, 1, 2, 3

    def __init__(self):
        self.regretSum = [0.0] * self.NUM_ACTIONS
        self.strategy = [0.0] * self.NUM_ACTIONS
        self.strategySum = [0.0] * self.NUM_ACTIONS
        self.oppStrategy = [0.34, 0.33, 0.33] # Arbitrary strategy for the opponent

    # Get current mixed strategy through regret-matching
    def getStrategy(self):
        normalizingSum = 0.0
        for i in range(self.NUM_ACTIONS):
            self.strategy[i] = max(self.regretSum[i], 0)
            normalizingSum += self.strategy[i]
        for i in range(self.NUM_ACTIONS):
            if (normalizingSum > 0):
                self.strategy[i] /= normalizingSum
            else:
                self.strategy[i] = 1.0 / self.NUM_ACTIONS
            self.strategySum[i] += self.strategy[i]
        return self.strategy

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

    # Training algorithm
    def train(self, iterations):
        actionUtility = [0.0] * self.NUM_ACTIONS
        for i in range(iterations):
            # Get the regret-matched strategies for each player
            strategy = self.getStrategy()
            myAction = self.getAction(strategy)
            oppAction = self.getAction(self.oppStrategy)
            # Compute the utilities of each possible action
            actionUtility[oppAction] = 0
            actionUtility[0 if oppAction == self.NUM_ACTIONS - 1 else oppAction + 1] = 1
            actionUtility[self.NUM_ACTIONS - 1 if oppAction ==  0 else oppAction - 1] = -1
            # Accumulate the regrets
            for j in range(self.NUM_ACTIONS):
                self.regretSum[j] += actionUtility[j] - actionUtility[myAction]


    # Compute the average strategy accross all iterations
    def getAverageStrategy(self):
        avgStrategy = [0.0] * self.NUM_ACTIONS
        normalizingSum = 0.0
        for i in range(self.NUM_ACTIONS):
            normalizingSum += self.strategySum[i]
        for i in range(self.NUM_ACTIONS):
            if (normalizingSum > 0):
                avgStrategy[i] = self.strategySum[i] / normalizingSum
            else:
                avgStrategy[i] = 1.0 / self.NUM_ACTIONS
        return avgStrategy

# trainer = RPSTrainer()
# trainer.train(1000000)
# print(trainer.getAverageStrategy())

# 2.5 Exercise: RPS Equilibrium

class Player:
    def __init__(self, numActions):
        self.regretSum = [0.0] * numActions
        self.strategy = [0.0] * numActions
        self.strategySum = [0.0] * numActions

class RPSTrainer2:
    # Definitions
    ROCK, PAPER, SCISSORS, NUM_ACTIONS = 0, 1, 2, 3

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
            actionUtility0[action1] = 0
            actionUtility0[0 if action1 == self.NUM_ACTIONS - 1 else action1 + 1] = 1
            actionUtility0[self.NUM_ACTIONS - 1 if action1 ==  0 else action1 - 1] = -1
            actionUtility1[action0] = 0
            actionUtility1[0 if action0 == self.NUM_ACTIONS - 1 else action0 + 1] = 1
            actionUtility1[self.NUM_ACTIONS - 1 if action0 ==  0 else action0 - 1] = -1
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

trainer = RPSTrainer2()
trainer.train(1000000)
print(trainer.getAverageStrategy(trainer.p0))
print(trainer.getAverageStrategy(trainer.p1))