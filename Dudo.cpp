#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>

class DudoTrainer {
public:
    // Dudo definitions
    const int NUM_SIDES = 6;
    const int NUM_ACTIONS = (2 * NUM_SIDES) + 1;
    const int DUDO = NUM_ACTIONS - 1;

    std::vector<int> claimNum{1,1,1,1,1,1,2,2,2,2,2,2};
    std::vector<int> claimRank{2,3,4,5,6,1,2,3,4,5,6,1};


    struct Node {
        // min, max are inclusive
        int MIN_ACTION;
        int MAX_ACTION;
        int NUM_ACTIONS;
        std::string infoSet;

        std::vector<double> regretSum;
        std::vector<double> strategy;
        std::vector<double> strategySum;

        Node(int minA, int maxA)
            : MIN_ACTION(minA),
              MAX_ACTION(maxA),
              NUM_ACTIONS(maxA - minA + 1),
              regretSum(NUM_ACTIONS, 0.0),
              strategy(NUM_ACTIONS, 0.0),
              strategySum(NUM_ACTIONS, 0.0) {}

        const std::vector<double>& getStrategy(double realizationWeight) {
            double normalizingSum = 0.0;

            for (int i = 0; i < NUM_ACTIONS; i++) {
                strategy[i] = std::max(regretSum[i], 0.0);
                normalizingSum += strategy[i];
            }

            for (int i = 0; i < NUM_ACTIONS; i++) {
                if (normalizingSum > 0) {
                    strategy[i] /= normalizingSum;
                }
                else {
                    strategy[i] = 1.0 / NUM_ACTIONS;
                }
                strategySum[i] += realizationWeight * strategy[i];
            }

            return strategy;
        }

        std::vector<double> getAverageStrategy() const {
            std::vector<double> avg(NUM_ACTIONS);
            double normalizingSum = 0.0;

            for (int i = 0; i < NUM_ACTIONS; i++) {
                normalizingSum += strategySum[i];
            }
            for (int i = 0; i < NUM_ACTIONS; i++) {
                if (normalizingSum > 0) {
                    avg[i] = strategySum[i] / normalizingSum;
                }
                else {
                    avg[i] = 1.0 / NUM_ACTIONS;
                }
            }
            return avg;
        }

        std::string toString() const {
            std::ostringstream out;
            out << std::setw(4) << infoSet << ": [";
            auto avg = getAverageStrategy();
            for (size_t i = 0; i < avg.size(); i++) {
                out << std::fixed << std::setprecision(5) << avg[i];
                if (i + 1 < avg.size()) out << " ";
            }
            out << "]";
            return out.str();
        }
    };

    std::unordered_map<uint64_t, Node> nodeMap;

    // Convert Dudo claim history to a string
    std::string claimHistoryToString(const std::vector<bool>& isClaimed) {
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
    uint64_t infoSetToInteger(int playerRoll, const std::vector<bool>& isClaimed) {
        uint64_t infoSetNum = playerRoll;
        for (int a = NUM_ACTIONS - 2; a >= 0; a--) {
            infoSetNum = (infoSetNum << 1) | (isClaimed[a] ? 1 : 0);
        }
        return infoSetNum;
    }

    double cfr(const std::vector<int>& nums,
               std::vector<bool>& history,
               double p0, double p1,
               int lastAction) {

        int plays = std::count(history.begin(), history.end(), true);
        int player = plays % 2;

        if (plays > 1 && lastAction == DUDO) {
            int lastClaim = -1;
            for (int a = DUDO - 1; a >= 0; a--) {
                if (history[a]) {
                    lastClaim = a;
                    break;
                }
            }

            int count = claimNum[lastClaim];
            int rank = claimRank[lastClaim];

            for (int n : nums) {
                if (n == 1 || n == rank) count--;
            }

            // std::cout << "DUDO CALLED: "<< std::endl;
            // for (int i = 0; i < history.size(); i++) {
            //     std::cout << history[i] << ", ";
            // }
            // std::cout << std::endl;
            // for (int i = 0; i < history.size()-1; i++) {
            //     if (history[i]) {
            //         std::cout << claimNum[i] << "*" << claimRank[i] << ",";
            //     }
            // }
            // std::cout << "DUDO" << std::endl;
            // std::cout << "player: " << player << ", num0: " << nums[player] << ", num1: " << nums[1-player] << std::endl;
            // std::cout << "winner: " << ((count <= 0) ? player : 1-player) << std::endl;

            return (count <= 0) ? 1.0 : -1.0;
        }

        std::string infoSetStr = std::to_string(nums[player]) + claimHistoryToString(history);
        uint64_t infoSetNum = infoSetToInteger(nums[player], history);

        Node* node;
        if (nodeMap.count(infoSetNum)) {
            node = &nodeMap.at(infoSetNum);
        } else {
            // Create the node for the infoSet if it doesn't exist
            int maxA = (plays > 0) ? DUDO : DUDO - 1;
            nodeMap.emplace(infoSetNum, Node(lastAction + 1, maxA));
            node = &nodeMap.at(infoSetNum);
            node->infoSet = infoSetStr;
        }

        const auto& strategy = node->getStrategy(player == 0 ? p0 : p1);
        std::vector<double> util(node->NUM_ACTIONS);
        double nodeUtil = 0.0;

        for (int a = 0; a < node->NUM_ACTIONS; a++) {
            history[node->MIN_ACTION + a] = true;

            if (player == 0)
                util[a] = -cfr(nums, history, p0 * strategy[a], p1, node->MIN_ACTION + a);
            else
                util[a] = -cfr(nums, history, p0, p1 * strategy[a], node->MIN_ACTION + a);

            history[node->MIN_ACTION + a] = false;
            nodeUtil += strategy[a] * util[a];
        }

        double cfWeight = (player == 0) ? p1 : p0;
        for (int a = 0; a < node->NUM_ACTIONS; a++) {
            node->regretSum[a] += cfWeight * (util[a] - nodeUtil);
        }

        return nodeUtil;
    }

    void resetStrategySums() {
        for (auto& [k , v] : nodeMap) {
            for (int i = 0; i < v.strategySum.size(); i++) {
                v.strategySum[i] = 0.0;
            }
        }
    }

    void train(int iterations) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> die(1, 6);

        double util = 0.0;

        int resetIndex = iterations / 5;

        for (int i = 0; i < iterations; i++) {
            // Reset strategySum after 20% of the iterations
            if (i == resetIndex) {
                resetStrategySums();
            }

            int d0 = die(gen);
            int d1 = die(gen);
            std::vector<bool> history(NUM_ACTIONS, false);

            util += cfr({d0, d1}, history, 1.0, 1.0, -1);

            if (i % 100 == 0) {
                std::cout << "Iteration: " << i << "\n";
                std::cout << "d0: " << d0 << ", d1: " << d1 << "\n";
                std::cout << "Average game value: " << util / i << "\n";
            }
        }

        std::cout << "Final average game value: " << util / iterations << "\n";
        std::cout << nodeMap.size() << " information sets \n";

    }
};


int main() {
    DudoTrainer trainer;
    trainer.train(10000);
    return 0;
}
