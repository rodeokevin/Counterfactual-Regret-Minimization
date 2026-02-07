#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstdlib>

class Node {
public:
    // Liar Die node definitions
    int numActions;
    std::vector<double> regretSum;
    std::vector<double> strategy;
    std::vector<double> strategySum;

    // utility value for each node
    double u = 0.0;
    // realization weights
    double pPlayer = 0.0;
    double pOpponent = 0.0;

    // Liar Die node constructor
    Node(int numActions)
        : numActions(numActions),
          regretSum(numActions, 0.0),
          strategy(numActions, 0.0), 
          strategySum(numActions, 0.0) {}
    
    // Get Liar Die node current mixed strategy through regret-matching
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

    // Get Liar Die node average mixed strategy
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
};

class LiarDieTrainer {
public:
    // Liar Die definitions
    static const int DOUBT = 0;
    static const int ACCEPT = 1;
    int sides;
    std::vector<std::vector<Node>> responseNodes;
    std::vector<std::vector<Node>> claimNodes;

    // Construct trainer and allocate player decision nodes
    // Currently using Node(0) to initially fill the 2D array. So for any indices that are not reassigned in the for loops, they will stay as Node objects with numActions = 0.
    // This should be fine because those indices should never be accessed during training as those are invalid game states.
    LiarDieTrainer(int sides):sides(sides) {
        responseNodes = std::vector<std::vector<Node>>(sides, std::vector<Node>(sides+1, Node(0)));
        for (int myClaim = 0; myClaim < sides; myClaim++) {
            for (int oppClaim = myClaim + 1; oppClaim <= sides; oppClaim++) {
                responseNodes[myClaim][oppClaim] = Node((oppClaim == 0 || oppClaim == sides) ? 1 : 2);
            }
        }
        claimNodes = std::vector<std::vector<Node>>(sides, std::vector<Node>(sides+1, Node(0)));
        for (int oppClaim = 0; oppClaim < sides; oppClaim++) {
            for (int roll = 1; roll <= sides; roll++) {
                claimNodes[oppClaim][roll] = Node(sides - oppClaim);
            }
        }
    }

    // Train with FSICFR
    void train(int iterations) {
        double gameValSum = 0.0;

        std::vector<double> regret(sides);
        std::vector<int> rollAfterAcceptingClaim(sides);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> die(1, sides);

        for (int iter = 0; iter < iterations; iter++) {
            // Initialize rolls and starting probabilities
            for (int i = 0; i < sides; i++) {
                rollAfterAcceptingClaim[i] = die(gen);
            }
            claimNodes[0][rollAfterAcceptingClaim[0]].pPlayer = 1;
            claimNodes[0][rollAfterAcceptingClaim[0]].pOpponent = 1;

            // Accumulate realization weights forward
            for (int oppClaim = 0; oppClaim <= sides; oppClaim++) {
                // Visit response nodes forward
                if (oppClaim > 0) {
                    for (int myClaim = 0; myClaim < oppClaim; myClaim++) {
                        Node& node = responseNodes[myClaim][oppClaim];
                        const std::vector<double>& actionProb = node.getStrategy();
                        if (oppClaim < sides) {
                            Node& nextNode = claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]];
                            nextNode.pPlayer += actionProb[1] * node.pPlayer;
                            nextNode.pOpponent += node.pOpponent;
                        }
                    }
                }
                // Visit claim nodes forward
                if (oppClaim < sides) {
                    Node& node = claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]];
                    const std::vector<double>& actionProb = node.getStrategy();
                    for (int myClaim = oppClaim + 1; myClaim <= sides; myClaim++) {
                        double nextClaimProb = actionProb[myClaim - oppClaim - 1];
                        if (nextClaimProb > 0) {
                            Node& nextNode = responseNodes[oppClaim][myClaim];
                            nextNode.pPlayer += node.pOpponent;
                            nextNode.pOpponent += nextClaimProb * node.pPlayer;
                        }
                    }
                }

            }
            // Backpropagate utilities, adjusting regrets and strategies
            for (int oppClaim = sides; oppClaim >= 0; oppClaim--) {
                // Visit claim nodes backward
                if (oppClaim < sides) {
                    Node& node = claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]];
                    std::vector<double>& actionProb = node.strategy;
                    node.u = 0.0;
                    for (int myClaim = oppClaim + 1; myClaim <= sides; myClaim++) {
                        int actionIndex = myClaim - oppClaim - 1;
                        Node& nextNode = responseNodes[oppClaim][myClaim];
                        double childUtil = - nextNode.u;
                        regret[actionIndex] = childUtil;
                        node.u += actionProb[actionIndex] * childUtil;
                    }
                    // accumulate counterfactual regret for each action for the node
                    for (int a = 0; a < actionProb.size(); a++) {
                        regret[a] -= node.u;
                        node.regretSum[a] += node.pOpponent * regret[a];
                    }
                    node.pPlayer = node.pOpponent = 0;
                }
                // Visit response nodes backward
                if (oppClaim > 0) {
                    for (int myClaim = 0; myClaim < oppClaim; myClaim++) {
                        Node& node = responseNodes[myClaim][oppClaim];
                        std::vector<double>& actionProb = node.strategy;
                        node.u = 0.0;
                        double doubtUtil = (oppClaim > rollAfterAcceptingClaim[myClaim]) ? 1 : -1;
                        regret[DOUBT] = doubtUtil;
                        node.u += actionProb[DOUBT] * doubtUtil;
                        if (oppClaim < sides) {
                            Node& nextNode = claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]];
                            regret[ACCEPT] = nextNode.u;
                            node.u += actionProb[ACCEPT] * nextNode.u;
                        }
                        for (int a = 0; a < actionProb.size(); a++) {
                            regret[a] -= node.u;
                            node.regretSum[a] += node.pOpponent * regret[a];
                        }
                        node.pPlayer = node.pOpponent = 0;
                    }
                }
            }
            // Reset strategy sums after half of training
            if (iter == iterations / 2) {
                for (auto& nodes : responseNodes) {
                    for (auto& node : nodes) {
                        for (int a = 0; a < node.strategySum.size(); a++) {
                            node.strategySum[a] = 0;
                        }
                    }
                }
                for (auto& nodes : claimNodes) {
                    for (auto& node : nodes) {
                        for (int a = 0; a < node.strategySum.size(); a++) {
                            node.strategySum[a] = 0;
                        }
                    }
                }
            }
            
            gameValSum += claimNodes[0][rollAfterAcceptingClaim[0]].u;
        }
        // Print resulting strategy
        std::cout << std::fixed << std::setprecision(5);
        for (int initialRoll = 1; initialRoll <= sides; initialRoll++) {
            std:: cout << "Initial claim policy with roll " << initialRoll << "\n";
            for (double& prob : claimNodes[0][initialRoll].getAverageStrategy()) {
                std::cout << prob << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\nOld Claim\tNew Claim\tAction Probabilities\n";
        // Response nodes
        for (int myClaim = 0; myClaim < sides; ++myClaim) {
            for (int oppClaim = myClaim + 1; oppClaim <= sides; ++oppClaim) {
                std::cout << myClaim << "\t\t" << oppClaim << "\t\t";
                const auto& strat = responseNodes[myClaim][oppClaim].getAverageStrategy();
                std::cout << '[';
                for (size_t i = 0; i < strat.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << strat[i];
                }
                std::cout << "]\n";
            }
        }

        std::cout << "\nOld Claim\tRoll\tAction Probabilities\n";
        // Claim nodes
        for (int oppClaim = 0; oppClaim < sides; ++oppClaim) {
            for (int roll = 1; roll <= sides; ++roll) {
                std::cout << oppClaim << "\t\t" << roll << '\t';
                const auto& strat = claimNodes[oppClaim][roll].getAverageStrategy();
                std::cout << '[';
                for (size_t i = 0; i < strat.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << strat[i];
                }
                std::cout << "]\n";
            }
        }

        double avgGameValue = gameValSum / iterations;
        std::cout << "Average game value: " << avgGameValue << "\n";
    }
};

int main(int argc, char* argv[]) {
    int iterations = 1000;
    int sides = 6;

    // Take a command line argument for number of iterations
    if (argc > 2) {
        sides = std::stoi(argv[1]);
        iterations = std::stoi(argv[2]);
    }

    LiarDieTrainer trainer(sides);
    trainer.train(iterations);
    return 0;
}