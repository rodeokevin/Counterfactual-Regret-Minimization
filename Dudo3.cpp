#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>

class Node {
public:
    // min, max are inclusive
    int minAction;
    int maxAction;
    int numActions;
    std::string infoSet;

    std::vector<double> regretSum;
    std::vector<double> strategy;
    std::vector<double> strategySum;

    bool processed = false;
    bool backProcessed = false;

    // utility value for each node
    double u = 0.0;
    // realization weights
    double pPlayer = 0.0;
    double pOpponent = 0.0;

    Node(int minA, int maxA)
        : minAction(minA),
            maxAction(maxA),
            numActions(maxA - minA + 1),
            regretSum(numActions, 0.0),
            strategy(numActions, 0.0),
            strategySum(numActions, 0.0) {}

    Node(std::string infoSetStr, int minA, int maxA)
        : minAction(minA),
            maxAction(maxA),
            numActions(maxA - minA + 1),
            infoSet(infoSetStr),
            regretSum(numActions, 0.0),
            strategy(numActions, 0.0),
            strategySum(numActions, 0.0) {}

    const std::vector<double>& getStrategy() {
        double normalizingSum = 0.0;

        for (int i = 0; i < numActions; i++) {
            strategy[i] = std::max(regretSum[i], 0.0);
            normalizingSum += strategy[i];
        }

        for (int i = 0; i < numActions; i++) {
            if (normalizingSum > 0) {
                strategy[i] /= normalizingSum;
            }
            else {
                strategy[i] = 1.0 / numActions;
            }
            strategySum[i] += pPlayer * strategy[i];
        }

        return strategy;
    }

    std::vector<double> getAverageStrategy() const {
        std::vector<double> avg(numActions);
        double normalizingSum = 0.0;

        for (int i = 0; i < numActions; i++) {
            normalizingSum += strategySum[i];
        }
        for (int i = 0; i < numActions; i++) {
            if (normalizingSum > 0) {
                avg[i] = strategySum[i] / normalizingSum;
            }
            else {
                avg[i] = 1.0 / numActions;
            }
        }
        return avg;
    }

    std::string toString() const {
        std::ostringstream out;
        out << infoSet << ": [";
        auto avg = getAverageStrategy();
        for (size_t i = 0; i < avg.size(); i++) {
            out << std::setw(4) << std::fixed << std::setprecision(5) << avg[i];
            if (i + 1 < avg.size()) out << " ";
        }
        out << "]";
        return out.str();
    }
};

class Dudo3Trainer {
public:
    // Dudo definitions
    const int NUM_SIDES = 6;
    const int NUM_ACTIONS = (2 * NUM_SIDES) + 1;
    const int DUDO = NUM_ACTIONS - 1;

    std::vector<int> claimNum{1,1,1,1,1,1,2,2,2,2,2,2};
    std::vector<int> claimRank{2,3,4,5,6,1,2,3,4,5,6,1};

    std::unordered_map<uint64_t, Node> nodeMap;

    // Convert Dudo claim history to a string
    std::string claimHistoryToString(const std::vector<bool>& isClaimed) const {
        std::string s;
        for (int a = 0; a < DUDO; a++) {
            if (isClaimed[a]) {
                if (!s.empty()) s += ",";
                s += std::to_string(claimNum[a]) + "*" + std::to_string(claimRank[a]);
            }
        }
        return s;
    }

    // Convert Dudo information set to an integer
    uint64_t infoSetToInteger(int playerRoll, const std::vector<bool>& isClaimed) const {
        uint64_t infoSetNum = playerRoll;
        for (int a = NUM_ACTIONS - 2; a >= 0; a--) {
            infoSetNum = (infoSetNum << 1) | (isClaimed[a] ? 1 : 0);
        }
        return infoSetNum;
    }

    // Constructs the trainer and allocates all possible nodes with memory of at most 3 claims
    Dudo3Trainer() {
        std::vector<bool> isClaimed(NUM_ACTIONS-1, false);
        // Possible dice ranks for each information set 
        for (int i = 1; i <= NUM_SIDES; i++) {
            // No claims in history (i.e. starting node)
            std::string infoSetStr = std::to_string(i) + claimHistoryToString(isClaimed);
            nodeMap.emplace(infoSetToInteger(i, isClaimed), Node(infoSetStr, 0, DUDO-1));
            // 1 claim in history
            for (int a = 0; a < NUM_ACTIONS-1; a++) {
                isClaimed[a] = true;
                std::string infoSetStr = std::to_string(i) + claimHistoryToString(isClaimed);
                nodeMap.emplace(infoSetToInteger(i, isClaimed), Node(infoSetStr, a+1, DUDO));
                // 2 claims in history
                for (int b = a+1; b < NUM_ACTIONS-1; b++) {
                    isClaimed[b] = true;
                    std::string infoSetStr = std::to_string(i) + claimHistoryToString(isClaimed);
                    nodeMap.emplace(infoSetToInteger(i, isClaimed), Node(infoSetStr, b+1, DUDO));
                    // 3 claims in history
                    for (int c = b+1; c < NUM_ACTIONS-1; c++) {
                        isClaimed[c] = true;
                        std::string infoSetStr = std::to_string(i) + claimHistoryToString(isClaimed);
                        nodeMap.emplace(infoSetToInteger(i, isClaimed), Node(infoSetStr, c+1, DUDO));
                        isClaimed[c] = false;
                    }
                    isClaimed[b] = false;
                }
                isClaimed[a] = false;
            }
        }
        
    }

    void train(int iterations) {
        double gameValSum = 0.0;

        std::vector<double> regret(NUM_SIDES);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> die(1, 6);

        for (int iter = 0; iter < iterations; iter++) {

            int d0 = die(gen);
            int d1 = die(gen);
            std::vector<bool> history(NUM_ACTIONS, false);

            if (iter % 100 == 0) {
                std::cout << "Iteration: " << iter << "\n";
            }
            

            std::vector<bool> isClaimed(NUM_ACTIONS-1, false);
            // Accumulate realization weights forward
            {
                std::vector<bool> emptyClaims(NUM_ACTIONS-1, false);
                // 0 claims in history (curr: d0, next: d1)
                Node &initialNode = nodeMap.at(infoSetToInteger(d0, emptyClaims));
                const std::vector<double>& actionProb = initialNode.getStrategy();
                initialNode.pPlayer = 1;
                initialNode.pOpponent = 1;

                
                // Iterate through next nodes
                for (int a = initialNode.minAction; a <= initialNode.maxAction; a++) { // Here the max action is not DUDO so we include it
                    isClaimed[a] = true;
                    Node &nextNode = nodeMap.at(infoSetToInteger(d1, isClaimed));
                    if (nextNode.processed == true) {
                        std::cout << "next node processed before current node\n";
                    }
                    nextNode.pPlayer += initialNode.pOpponent;
                    nextNode.pOpponent += actionProb.at(a - initialNode.minAction) * initialNode.pPlayer;
                    isClaimed[a] = false;
                }
                initialNode.processed = true;
            }
            

            // 1 claim in history (curr: d1, next: d0)
            for (int i = 0; i < NUM_ACTIONS-1; i++) {
                isClaimed[i] = true;
                Node &node = nodeMap.at(infoSetToInteger(d1, isClaimed));
                const std::vector<double>& actionProb = node.getStrategy();
                // Iterate through next nodes
                for (int a = node.minAction; a < node.maxAction; a++) {
                    isClaimed[a] = true;
                    Node &nextNode = nodeMap.at(infoSetToInteger(d0, isClaimed));
                    if (nextNode.processed == true) {
                        std::cout << "next node processed before current node\n";
                    }
                    nextNode.pPlayer += node.pOpponent;
                    nextNode.pOpponent += actionProb.at(a - node.minAction) * node.pPlayer;
                    isClaimed[a] = false;
                }
                isClaimed[i] = false;
                node.processed = true;
            }

            // 2 claims in history (curr: d0, next: d1)
            for (int i = 0; i < NUM_ACTIONS-1; i++) {
                isClaimed[i] = true;
                for (int j = i+1; j < NUM_ACTIONS-1; j++) {
                    isClaimed[j] = true;
                    Node &node = nodeMap.at(infoSetToInteger(d0, isClaimed));
                    const std::vector<double>& actionProb = node.getStrategy();
                    // Iterate through next nodes
                    for (int a = node.minAction; a < node.maxAction; a++) {
                        isClaimed[a] = true;
                        Node &nextNode = nodeMap.at(infoSetToInteger(d1, isClaimed));
                        if (nextNode.processed == true) {
                            std::cout << "next node processed before current node\n";
                            std::cout << "current - d0: " << d0 << ", claims: " << i << ", " << j << "\n";
                            std::cout << "next - d1: " << d1 << ", claims: " << i << ", " << j << ", " << a << "\n"; 
                        }
                        nextNode.pPlayer += node.pOpponent;
                        nextNode.pOpponent += actionProb.at(a - node.minAction) * node.pPlayer;
                        isClaimed[a] = false;
                    }
                    node.processed = true;
                    isClaimed[j] = false;
                }
                isClaimed[i] = false;
            }

            // 3 claims in history (curr: d1, next: d0, or curr: d0, next: d1 if claim 0 hasn't been made and d0 != d1)
            for (int i = 0; i < NUM_ACTIONS-1; i++) {
                isClaimed[i] = true;
                for (int j = i+1; j < NUM_ACTIONS-1; j++) {
                    isClaimed[j] = true;
                    for (int k = j+1; k < NUM_ACTIONS-1; k++) {
                        isClaimed[k] = true;

                        Node &node = nodeMap.at(infoSetToInteger(d1, isClaimed));
                        if (node.processed == true) {
                            std::cout << "This node was already processed\n";
                            std::cout << "current - d1: " << d1 << ", claims: " << i << ", " << j << ", " << k << "\n";
                        }
                        // std::cout << "current: " << i << ", " << j << ", " << k << "\n";
                        const std::vector<double>& actionProb = node.getStrategy();
                        // Iterate through next nodes
                        for (int a = node.minAction; a < node.maxAction; a++) {
                            // std::cout << "next: " << j << ", " << k << ", " << a << "\n";
                            isClaimed[i] = false;
                            isClaimed[a] = true;
                            Node &nextNode = nodeMap.at(infoSetToInteger(d0, isClaimed));
                            if (nextNode.processed == true) {
                                std::cout << "next node processed before current node\n";
                                std::cout << "current - d1: " << d1 << ", claims: " << i << ", " << j << ", " << k << "\n";
                                std::cout << "next - d0: " << d0 << ", claims: " << j << ", " << k << ", " << a << "\n";    
                            }
                            nextNode.pPlayer += node.pOpponent;
                            nextNode.pOpponent += actionProb.at(a - node.minAction) * node.pPlayer;
                            isClaimed[a] = false;
                            isClaimed[i] = true;
                        }
                        node.processed = true;

                        // (curr: d0, next: d1)
                        if (isClaimed[0] == false && d0 != d1) {
                            Node &node = nodeMap.at(infoSetToInteger(d0, isClaimed));
                            if (node.processed == true) {
                                std::cout << "This node was already processed\n";
                                std::cout << "current - d0: " << d0 << ", claims: " << i << ", " << j << ", " << k << "\n";
                            }
                            const std::vector<double>& actionProb = node.getStrategy();
                            // Iterate through next nodes
                            for (int a = node.minAction; a < node.maxAction; a++) {
                                // std::cout << "next: " << j << ", " << k << ", " << a << "\n";
                                isClaimed[i] = false;
                                isClaimed[a] = true;
                                Node &nextNode = nodeMap.at(infoSetToInteger(d1, isClaimed));
                                if (nextNode.processed == true) {
                                    std::cout << "\n";
                                    std::cout << "next node processed before current node\n";
                                    std::cout << "current - d0: " << d0 << ", claims: " << i << ", " << j << ", " << k << "\n";
                                    std::cout << "next - d1: " << d1 << ", claims: " << j << ", " << k << ", " << a << "\n";
                                }
                                // else {

                                //     std::cout << "\n";
                                //     std::cout << "everything is good!\n";
                                // }
                                nextNode.pPlayer += node.pOpponent;
                                nextNode.pOpponent += actionProb.at(a - node.minAction) * node.pPlayer;
                                isClaimed[a] = false;
                                isClaimed[i] = true;
                            }
                            node.processed = true;
                        }
                        isClaimed[k] = false;
                    }
                    isClaimed[j] = false;
                }
                isClaimed[i] = false;
            }


            // Backpropagate utilities, adjusting regrets and strategies
            // 3 claims in history (curr: d1, next: d0, or curr: d0, next: d1 if claim 0 hasn't been made and d0 != d1)
            for (int i = NUM_ACTIONS-2; i >= 0; i--) {
                isClaimed[i] = true;
                for (int j = i-1; j >= 0; j--) {
                    isClaimed[j] = true;
                    for (int k = j-1; k >= 0; k--) {
                        isClaimed[k] = true;

                        Node &node = nodeMap.at(infoSetToInteger(d1, isClaimed));
                        if (node.backProcessed == true) {
                            std::cout << "This node was already backProcessed\n";
                            std::cout << "current - d1: " << d1 << ", claims: " << i << ", " << j << ", " << k << "\n";
                        }
                        // std::cout << "current: " << i << ", " << j << ", " << k << "\n";
                        const std::vector<double>& actionProb = node.getStrategy();
                        // Iterate through next nodes
                        for (int a = node.minAction; a < node.maxAction; a++) {
                            // std::cout << "next: " << j << ", " << k << ", " << a << "\n";
                            isClaimed[i] = false;
                            isClaimed[a] = true;
                            Node &nextNode = nodeMap.at(infoSetToInteger(d0, isClaimed));
                            if (nextNode.processed == true) {
                                std::cout << "next node processed before current node\n";
                                std::cout << "current - d1: " << d1 << ", claims: " << i << ", " << j << ", " << k << "\n";
                                std::cout << "next - d0: " << d0 << ", claims: " << j << ", " << k << ", " << a << "\n";    
                            }
                            nextNode.pPlayer += node.pOpponent;
                            nextNode.pOpponent += actionProb.at(a - node.minAction) * node.pPlayer;
                            isClaimed[a] = false;
                            isClaimed[i] = true;
                        }
                        node.processed = true;

                        // (curr: d0, next: d1)
                        if (isClaimed[0] == false && d0 != d1) {
                            Node &node = nodeMap.at(infoSetToInteger(d0, isClaimed));
                            if (node.processed == true) {
                                std::cout << "This node was already processed\n";
                                std::cout << "current - d0: " << d0 << ", claims: " << i << ", " << j << ", " << k << "\n";
                            }
                            const std::vector<double>& actionProb = node.getStrategy();
                            // Iterate through next nodes
                            for (int a = node.minAction; a < node.maxAction; a++) {
                                // std::cout << "next: " << j << ", " << k << ", " << a << "\n";
                                isClaimed[i] = false;
                                isClaimed[a] = true;
                                Node &nextNode = nodeMap.at(infoSetToInteger(d1, isClaimed));
                                if (nextNode.processed == true) {
                                    std::cout << "\n";
                                    std::cout << "next node processed before current node\n";
                                    std::cout << "current - d0: " << d0 << ", claims: " << i << ", " << j << ", " << k << "\n";
                                    std::cout << "next - d1: " << d1 << ", claims: " << j << ", " << k << ", " << a << "\n";
                                }
                                // else {

                                //     std::cout << "\n";
                                //     std::cout << "everything is good!\n";
                                // }
                                nextNode.pPlayer += node.pOpponent;
                                nextNode.pOpponent += actionProb.at(a - node.minAction) * node.pPlayer;
                                isClaimed[a] = false;
                                isClaimed[i] = true;
                            }
                        }

                        node.processed = true;
                        isClaimed[k] = false;
                    }
                    isClaimed[j] = false;
                }
                isClaimed[i] = false;
            }






            // Reset strategy sums after half of training
            if (iter == iterations / 2) {
                for (auto& [_, node] : nodeMap) {
                    for (int a = 0; a < node.strategySum.size(); a++) {
                        node.strategySum[a] = 0;
                    }
                }
            }

            std::vector<bool> emptyClaims(NUM_ACTIONS-1, false);
            gameValSum += nodeMap.at(infoSetToInteger(d0, emptyClaims)).u;
            std::cout << "d0: " << d0 << ", d1: " << d1 << "\n";
            // std::cout << nodeMap.at(intialNodeKey).infoSet << "\n";




        }

        

        
        int numProcessed = 0;
        std::ofstream out("output.txt");
        for (const auto& [_, node] : nodeMap) {
            out << node.toString() << "\n";
            out << "pPlayer: " << node.pPlayer << " pOpponent: " << node.pOpponent << "\n";
            out << "minAction: " << node.minAction << ", maxAction: " << node.maxAction << "\n";
            if (node.processed) {
                numProcessed += 1;
                out << "processed\n";
            }
            else {
                if (node.pOpponent != 0.0 || node.pPlayer != 0.0) {
                    out << "something's awry here\n";
                }
            }
            
        }
        std::cout << numProcessed << " nodes processed in forward prop\n";
           
    }
};


int main() {
    Dudo3Trainer trainer;
    std::cout << trainer.nodeMap.size() << " information sets \n";
    trainer.train(1);
    return 0;
}
