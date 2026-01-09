import numpy as np

# 3.4 Worked Example: Kuhn Poker

# Kuhn Trainer
class KuhnTrainer:
    # Kuhn poker definitions
    def __init__(self):
        self.PASS = 0
        self.BET = 1
        self.NUM_ACTIONS = 2
        self.nodeMap = {}
    
    # Information set node class definition
    class Node:
        def __init__(self, numActions):
            self.NUM_ACTIONS = numActions
            self.infoSet = ""
            self.regretSum = np.zeros(numActions)
            self.strategy = np.zeros(numActions)
            self.strategySum = np.zeros(numActions)
        
        def getStrategy(self, realizationWeight):
            strategy = np.clip(self.regretSum, a_min=0, a_max=None)
            normalizingSum = sum(strategy)
            if (normalizingSum > 0):
                strategy /= normalizingSum
            else:
                strategy = np.repeat(1.0 / self.NUM_ACTIONS, self.NUM_ACTIONS)
            self.strategySum += realizationWeight * strategy
            return strategy

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
    
    # Train Kuhn poker
    def train(self, iterations):
        cards = np.array([0,1,2])
        util = 0.0
        for _ in range(iterations):
            np.random.shuffle(cards)
            util += self.cfr(cards, "", 1, 1)
        print("Average game value: " + str(util / iterations))
        for key in sorted(self.nodeMap):
            print(self.nodeMap[key].toString())
    
    # Counterfactual regret minimization iteration
    def cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player
        # Return payoff for terminal states
        if (plays > 1):
            terminalPass = history[plays - 1] == 'p'
            doubleBet = history[-2:] == "bb"
            isPlayerCardHigher = cards[player] > cards[opponent]
            if (terminalPass):
                if (history == "pp"):
                    return 1 if isPlayerCardHigher else -1
                else:
                    return 1
            elif (doubleBet):
                return 2 if isPlayerCardHigher else -2
        infoSet = str(cards[player]) + history
        # Get information set node or create it if nonexistant
        if (infoSet in self.nodeMap):
            node = self.nodeMap[infoSet]
        else:
            node = self.Node(self.NUM_ACTIONS)
            node.infoSet = infoSet
            self.nodeMap[infoSet] = node
        # For each action, recursively call cfr with additional history and probability
        node.strategy = node.getStrategy(p0 if player == 0 else p1)
        util = np.zeros(self.NUM_ACTIONS)
        nodeUtil = 0
        for a in range(self.NUM_ACTIONS):
            nextHistory = history + ("p" if a == 0 else "b")
            if (player == 0):
                util[a] = -self.cfr(cards, nextHistory, p0 * node.strategy[a], p1)
            else:
                util[a] = -self.cfr(cards, nextHistory, p0, p1 * node.strategy[a])
            # nodeUtil += node.strategy[a] * util[a]
        nodeUtil = np.sum(node.strategy * util)
        # For each action, compute and accumulate counterfactual regret

        # for a in range(self.NUM_ACTIONS):
        #     regret = util[a] - nodeUtil
        #     node.regretSum[a] += (p1 if player == 0 else p0) * regret
        node.regretSum += (p1 if player == 0 else p0) * (util - nodeUtil)
        return nodeUtil
    
trainer = KuhnTrainer()
trainer.train(1_000_000)