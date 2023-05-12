//
// Created by spook on 26.04.23.
//

#include "BoundUnfolder.h"
#include "storm/logic/BoundedUntilFormula.h"
#include "storm/exceptions/NotSupportedException.h"
#include <queue>

namespace storm {
    namespace transformer {
        template<typename ValueType>
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> BoundUnfolder<ValueType>::unfold(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPOMDP, const storm::logic::QuantileFormula& formula) {
            // Check formula
            assert(!formula.isMultiDimensional());
            STORM_LOG_THROW(formula.isProbabilityOperatorFormula() && formula.getSubformula().isBoundedUntilFormula() && formula.getSubformula().asBoundedUntilFormula().getLeftSubformula().isTrueFormula(), storm::exceptions::NotSupportedException, "Unexpected formula type of formula " << formula);

            // Grab reward model
            auto temp = std::set<std::string>();
            formula.gatherReferencedRewardModels(temp);
            assert(temp.size() == 1);
            auto rewModel = originalPOMDP->getRewardModel(temp.begin());
            STORM_LOG_THROW(rewModel.hasStateActionRewards(), storm::exceptions::NotSupportedException, "Only state action rewards are currently supported.");

            // Grab bound
            ValueType bound = formula.getSubformula().asBoundedUntilFormula().getUpperBound();

            // Grab matrix (mostly for coding convenience to just have it in a variable here)
            auto& ogMatrix = originalPOMDP->getTransitionMatrix();

            // Transformation information + variables (remove non-necessary ones later)
            auto stateEpochToNewState  = std::map<std::pair<uint_fast64_t, ValueType>, uint_fast64_t>();
            auto newStateToStateEpoch = std::map<uint_fast64_t, std::pair<uint_fast64_t, ValueType>>();
            uint_fast64_t nextNewStateIndex = 2;
            std::queue<std::pair<uint_fast64_t, ValueType>> processingQ; // queue does BFS, if DFS is desired, change to stack

            // Information for unfolded model
            auto transitions = std::vector<std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>>();
            auto observations = std::vector<uint32_t>();
            uint_fast64_t entryCount = 0;
            uint_fast64_t choiceCount = 0;

            // Special sink states: 0 is =), 1 is =(
            transitions.push_back(std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>(std::map<std::pair<uint_fast64_t, ValueType>, ValueType>(), 1));
            transitions [0][0][0] = storm::utility::one<ValueType>();
            entryCount++;
            choiceCount++;
            transitions.push_back(std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>(std::map<std::pair<uint_fast64_t, ValueType>, ValueType>(), 1));
            transitions [1][0][1] = storm::utility::one<ValueType>();
            entryCount++;
            choiceCount++;

            // Create init state of unfolded model
            assert(originalPOMDP->getInitialStates().getNumberOfSetBits() == 1);
            uint_fast64_t initState = originalPOMDP->getInitialStates().getNextSetIndex(0);
            auto initEpochState = std::make_pair(initState, bound);
            processingQ.push(initEpochState);
            auto numberOfActions = ogMatrix.getRowGroupSize(initState);
            transitions.push_back(std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>(std::map<std::pair<uint_fast64_t, ValueType>, ValueType>(), numberOfActions));
            stateEpochToNewState[initEpochState] = nextNewStateIndex;
            newStateToStateEpoch[nextNewStateIndex] = initEpochState;
            nextNewStateIndex++;
            entryCount++;
            choiceCount++;


            while (!processingQ.empty()) {
                auto currentEpochState = processingQ.pop();
                // TODO add transitions to special states from states with the goal observation (also see further below)
                uint_fast64_t rowGroupStart = ogMatrix.getRowGroupIndices()[currentEpochState.first];
                uint_fast64_t rowGroupSize = ogMatrix.getRowGroupSize(currentEpochState.first);
                for (auto row = rowGroupStart; row < rowGroupStart + rowGroupSize; row++) {
                    choiceCount++;
                    auto actionIndex = row - rowGroupStart;
                    auto reward = rewModel.getStateActionReward(row);
                    for (auto entry : ogMatrix.getRow(row)) {
                        // Calculate unfolded successor state
                        uint_fast64_t oldSuccState = entry.getColumn();
                        ValueType epoch;
                        if (currentEpochState.second >= reward) {
                            epoch = currentEpochState.second - rewModel.getStateActionReward(row);
                        } else {
                            epoch = storm::utility::infinity<ValueType>(); // TODO does this even work for doubles? maybe come up with other denotation of bottom
                        }
                        auto stateEpochSucc = std::make_pair(oldSuccState, epoch);
                        // If unfolded successor does not exist yet, create it + add it to processing queue
                        if (stateEpochToNewState.find(stateEpochSucc) == stateEpochToNewState.end()){
                            stateEpochToNewState[stateEpochSucc] = nextNewStateIndex;
                            newStateToStateEpoch[nextNewStateIndex] = stateEpochSucc;
                            numberOfActions = ogMatrix.getRowGroupSize(oldSuccState);
                            transitions.push_back(std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>(std::map<std::pair<uint_fast64_t, ValueType>, ValueType>(), numberOfActions));
                            nextNewStateIndex++;
                            // TODO transitions to special states here if its a goal state + only put in processing queue if it isn't
                            // TODO what about states with epoch = bottom? we don't really need to pursue them any further, do we? so maybe transition to =( immediately? don't know if that might be a hindrance for multi-cost bounded stuff later tho

                            // TODO maybe extra handling of sink states in og mdp: instead of making multiple copies where epoch != bottom that eventually lead to a copy where epoch = bottom, make immediate transition to =(
                            processingQ.push(stateEpochSucc);
                        }
                        // Add transition
                        uint_fast64_t newSuccState = stateEpochToNewState[stateEpochSucc];
                        transitions[currentEpochState][actionIndex][newSuccState] = entry.getValue();
                        entryCount++;
                    }
                }
            }
            // TODO do we want the new states to be ordered a certain way?
            // Observations
            for (uint_fast64_t i = 0; i < nextNewStateIndex; i++){
                // TODO extra case for special states =) and =(. are the observation numbers dependent on eg state labeling?
                observations.push_back(originalPOMDP->getObservation(newStateToStateEpoch[i].first));
            }

            // Lets get building (taken from beliefmdpexplorer + adapted)
            storm::storage::SparseMatrixBuilder<ValueType> builder(choiceCount, nextNewStateIndex, entryCount, true, true, nextNewStateIndex);
            uint_fast64_t nextMatrixRow = 0;
            for (uint_fast64_t state = 0; state < transitions.size(); state++){
                builder.newRowGroup(nextMatrixRow);
                for (auto action = 0; action < transitions[state].size(); action++){
                    for (auto const &entry : transitions[state][action]) {
                        builder.addNextValue(nextMatrixRow, entry.first, entry.second);
                        nextMatrixRow++;
                    }
                }
            }
            auto unfoldedTransitionMatrix = builder.build();


            return std::shared_ptr<storm::models::sparse::Pomdp<ValueType>>();
        }
    }
}