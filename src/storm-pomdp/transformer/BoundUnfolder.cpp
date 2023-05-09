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
            // check everything has the right format etc
            assert(!formula.isMultiDimensional());
            STORM_LOG_THROW(formula.isProbabilityOperatorFormula() && formula.getSubformula().isBoundedUntilFormula(), storm::exceptions::NotSupportedException, "Unexpected formula type of formula " << formula);

            auto temp = std::set<std::string>();
            formula.gatherReferencedRewardModels(temp);
            assert(temp.size() == 1);
            auto rewModel = originalPOMDP->getRewardModel(temp.begin());
            STORM_LOG_THROW(rewModel.hasStateActionRewards(), storm::exceptions::NotSupportedException, "Only state action rewards are currently supported.");

            ValueType bound = formula.getSubformula().asBoundedUntilFormula().getUpperBound();
            assert(originalPOMDP->getInitialStates().getNumberOfSetBits() == 1);

            uint_fast64_t initState = originalPOMDP->getInitialStates().getNextSetIndex(0);
            auto ogMatrix = originalPOMDP->getTransitionMatrix();

            // information we need to build the model (remove non-necessary ones later)
            auto stateEpochToNewState  = std::map<std::pair<uint_fast64_t, ValueType>, uint_fast64_t>();
            auto newStateToStateEpoch = std::map<uint_fast64_t, std::pair<uint_fast64_t, ValueType>>();
            auto transitions = std::vector<std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>>(); // per state per action per succState, one probability // TODO rethink or convert to matrix rows after while loop
            auto observations = std::vector<uint32_t>();

            // for the matrix
            uint_fast64_t entryCount = 0;
            uint_fast64_t choiceCount = 0;

            // prep
            std::queue<std::pair<uint_fast64_t, ValueType>> processingQ; // queue does BFS, if DFS is desired, change to stack

            auto initEpochState = std::make_pair(initState, bound);
            processingQ.push(initEpochState);
            auto numberOfActions = originalPOMDP->getTransitionMatrix().getRowGroupSize(initState);
            transitions.push_back(std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>(numberOfActions));
            stateEpochToNewState[initEpochState] = 0;
            newStateToStateEpoch [0] = initEpochState;
            uint_fast64_t nextNewStateIndex = 1;


            while (!processingQ.empty()) {
                auto currentEpochState = processingQ.pop();
                uint_fast64_t rowGroupStart = ogMatrix.getRowGroupIndices()[currentEpochState.first];
                uint_fast64_t rowGroupSize = ogMatrix.getRowGroupSize(currentEpochState.first);
                for (auto row = rowGroupStart; row < rowGroupStart + rowGroupSize; row++) {
                    choiceCount++;
                    auto actionIndex = row - rowGroupStart;
                    for (auto entry : ogMatrix.getRow(row)) {
                        uint_fast64_t oldSuccState = entry.getColumn();
                        ValueType epoch;
                        if (currentEpochState.second >= rewModel.getStateActionReward(row)) {
                            epoch = currentEpochState.second - rewModel.getStateActionReward(row);
                        } else {
                            epoch = storm::utility::infinity<ValueType>(); // TODO does this even work for doubles? maybe come up with other denotation of bot
                        }
                        auto stateEpochSucc = std::make_pair(oldSuccState, epoch);
                        if (stateEpochToNewState.find(stateEpochSucc) == stateEpochToNewState.end()){
                            stateEpochToNewState[stateEpochSucc] = nextNewStateIndex;
                            newStateToStateEpoch[nextNewStateIndex] = stateEpochSucc;
                            numberOfActions = originalPOMDP->getTransitionMatrix().getRowGroupSize(oldSuccState);
                            transitions.push_back(std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>(numberOfActions));
                            processingQ.push(stateEpochSucc);
                            nextNewStateIndex++;
                        }
                        uint_fast64_t newSuccState = stateEpochToNewState[stateEpochSucc];
                        transitions[currentEpochState][actionIndex][newSuccState] = entry.getValue();
                        entryCount++;
                    }
                }
            }
            // TODO do we want the new states to be ordered a certain way?
            for (uint_fast64_t i = 0; i < nextNewStateIndex; i++){
                observations.push_back(originalPOMDP->getObservation(newStateToStateEpoch[i].first));
            }

            // lets get building (taken from beliefmdpexplorer + adapted)
            storm::storage::SparseMatrixBuilder<ValueType> builder(choiceCount, nextNewStateIndex, entryCount, true, true, nextNewStateIndex);
            uint_fast64_t nextGroupIndex = 0;
            for (uint_fast64_t state = 0; state < transitions.size(); state++){
                for (auto action = 0; action < transitions[state].size(); action++){
                    for (auto const &entry : )
                }
            }
            for (uint64_t groupIndex = 0; groupIndex < exploredChoiceIndices.size() - 1; ++groupIndex) {
                uint64_t rowIndex = exploredChoiceIndices[groupIndex];
                uint64_t groupEnd = exploredChoiceIndices[groupIndex + 1];
                builder.newRowGroup(rowIndex);
                for (; rowIndex < groupEnd; ++rowIndex) {
                    for (auto const &entry : exploredMdpTransitions[rowIndex]) {
                        if (stateRemapping.find(entry.first) == stateRemapping.end()) {
                            builder.addNextValue(rowIndex, entry.first, entry.second);
                        } else {
                            builder.addNextValue(rowIndex, stateRemapping[entry.first], entry.second);
                        }
                    }
                }
            }
            auto mdpTransitionMatrix = builder.build();


            return std::shared_ptr<storm::models::sparse::Pomdp<ValueType>>();
        }
    }
}