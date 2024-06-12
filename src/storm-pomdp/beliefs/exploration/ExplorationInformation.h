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
    std::unordered_set<BeliefId> frontierBeliefs;
    BeliefId initialBeliefId;
    ExplorationQueue queue;
    // frontier beliefs as method instead of member
};
template<typename BeliefMdpValueType, typename BeliefType>
using StandardExplorationInformation = ExplorationInformation<BeliefMdpValueType, BeliefType>;

template<typename BeliefMdpValueType, typename BeliefType>
using RewardAwareExplorationInformation = ExplorationInformation<BeliefMdpValueType, BeliefType, std::vector<BeliefMdpValueType>>;
}  // namespace storm::pomdp::beliefs