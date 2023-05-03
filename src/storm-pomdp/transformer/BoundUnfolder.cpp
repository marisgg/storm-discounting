//
// Created by spook on 26.04.23.
//

#include "BoundUnfolder.h"
#include <queue>

namespace storm {
    namespace transformer {
        template<typename ValueType>
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> BoundUnfolder<ValueType>::unfold(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPOMDP, const storm::logic::QuantileFormula& formula) {
            // check everything has the right format etc
            assert(!formula.isMultiDimensional());
            auto temp = std::set<std::string>();
            formula.gatherReferencedRewardModels(temp);
            assert(temp.size() == 1);
            ValueType bound = storm::utility::one<ValueType>(); // TODO change to actual bound once i know how to do that
            assert(originalPOMDP->getInitialStates().getNumberOfSetBits() == 1);
            uint_fast64_t initState = originalPOMDP->getInitialStates().getNextSetIndex(0);
            auto ogMatrix = originalPOMDP->getTransitionMatrix();

            // information we need to build the model
            auto stateEpochToTransIndex  = std::map<std::pair<uint_fast64_t, ValueType>, uint_fast64_t>();
            auto transIndexToStateEpoch = std::map<uint_fast64_t, std::pair<uint_fast64_t, ValueType>>();
            auto transitions = std::vector<std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>>(); // per state per action per succState, one probability
            auto observations = std::vector<uint32_t>();

            // prep
            std::queue<std::pair<uint_fast64_t, ValueType>> processingQ;

            auto initEpochState = std::make_pair(initState, bound);
            processingQ.push(initEpochState);
            auto numberOfActions = originalPOMDP->getTransitionMatrix().getRowGroupSize(initState);
            transitions.push_back(std::vector<std::map<std::pair<uint_fast64_t, ValueType>, ValueType>>(numberOfActions));
            stateEpochToTransIndex[initEpochState] = 0;
            transIndexToStateEpoch [0] = initEpochState;

            while (!processingQ.empty()) {
                auto currentEpochState = processingQ.pop();
                auto rowGroupStart = ogMatrix.getRowGroupIndices()[currentEpochState.first];
                auto rowGroupSize = ogMatrix.getRowGroupSize(currentEpochState.first);
                for (auto row = rowGroupStart; row < rowGroupStart + rowGroupSize; row++) {
                    auto actionIndex = row - rowGroupStart;
                    for (auto entry : ogMatrix.getRow(row)) {
                        //TODO continue here
                    }
                }
            }

            return std::shared_ptr<storm::models::sparse::Pomdp<ValueType>>();
        }
    }
}