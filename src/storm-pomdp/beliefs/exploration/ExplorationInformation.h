#pragma once
#include "storm-pomdp/beliefs/exploration/BeliefExplorationMatrix.h"
#include "storm-pomdp/beliefs/exploration/ExplorationQueue.h"
#include "storm-pomdp/beliefs/storage/BeliefCollector.h"
#include "storm-pomdp/beliefs/utility/types.h"

namespace storm::pomdp::beliefs {
template<typename BeliefMdpValueType, typename BeliefType, typename... ExtraTransitionData>
struct ExplorationInformation {
    BeliefExplorationMatrix<BeliefMdpValueType, ExtraTransitionData...> matrix;
    std::vector<BeliefMdpValueType> actionRewards;
    storm::pomdp::beliefs::BeliefCollector<BeliefType> discoveredBeliefs;
    std::unordered_map<BeliefId, BeliefStateType> exploredBeliefs;
    std::unordered_map<BeliefId, BeliefMdpValueType> terminalBeliefValues;
    BeliefId initialBeliefId;
    ExplorationQueue queue;

    std::unordered_set<BeliefId> getFrontierBeliefs() const {
        std::unordered_set<BeliefId> resFrontierBeliefs;
        for (uint64_t id = 0; id < discoveredBeliefs.getNumberOfBeliefIds(); id++) {
            if (exploredBeliefs.count(id) == 0 && terminalBeliefValues.count(id) == 0) {
                resFrontierBeliefs.insert(id);
            }
        }
        return resFrontierBeliefs;
    }
};
template<typename BeliefMdpValueType, typename BeliefType>
using StandardExplorationInformation = ExplorationInformation<BeliefMdpValueType, BeliefType>;

template<typename BeliefMdpValueType, typename BeliefType>
using RewardAwareExplorationInformation = ExplorationInformation<BeliefMdpValueType, BeliefType, std::vector<BeliefMdpValueType>>;
}  // namespace storm::pomdp::beliefs