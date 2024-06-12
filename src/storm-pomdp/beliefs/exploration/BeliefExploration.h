#pragma once

#include <functional>
#include <optional>
#include <set>

#include "storm-pomdp/beliefs/exploration/ExplorationInformation.h"

#include "storm-pomdp/beliefs/exploration/FirstStateNextStateGenerator.h"
#include "storm-pomdp/beliefs/utility/types.h"

#include "storm/utility/OptionalRef.h"
#include "storm/utility/vector.h"

namespace storm::pomdp::beliefs {

template<typename BeliefType>
class FreudenthalTriangulationBeliefAbstraction;

template<typename BeliefMdpValueType, typename PomdpType, typename BeliefType>
class RewardBoundedBeliefSplitter;

template<typename BeliefMdpValueType, typename PomdpType, typename BeliefType>
class BeliefExploration {
   public:
    using TerminationCallback = std::function<bool()>;
    using TerminalBeliefCallback = std::function<std::optional<BeliefMdpValueType>(BeliefType const&)>;

    BeliefExploration(PomdpType const& pomdp);

    template<typename InfoType>
    InfoType initializeExploration(ExplorationQueueOrder const explorationQueueOrder = ExplorationQueueOrder::Unordered) {
        InfoType info;
        info.queue.changeOrder(explorationQueueOrder);
        info.initialBeliefId = info.discoveredBeliefs.addBelief(firstStateNextStateGenerator.computeInitialBelief());
        info.queue.push(info.initialBeliefId);
        return info;
    }

    void resumeExploration(StandardExplorationInformation<BeliefMdpValueType, BeliefType>& info, TerminalBeliefCallback const& terminalBeliefCallback = {},
                           TerminationCallback const& terminationCallback = {}, storm::OptionalRef<std::string const> rewardModelName = {},
                           storm::OptionalRef<FreudenthalTriangulationBeliefAbstraction<BeliefType>> abstraction = {});

    void resumeRewardAwareExploration(RewardAwareExplorationInformation<BeliefMdpValueType, BeliefType>& info,
                                      TerminalBeliefCallback const& terminalBeliefCallback, TerminationCallback const& terminationCallback,
                                      RewardBoundedBeliefSplitter<BeliefMdpValueType, PomdpType, BeliefType> rewardSplitter,
                                      storm::OptionalRef<FreudenthalTriangulationBeliefAbstraction<BeliefType>> abstraction);

   private:
    template<typename InfoType, typename NextStateHandleType>
    bool performExploration(InfoType& info, NextStateHandleType&& exploreNextStates, TerminalBeliefCallback const& terminalBeliefCallback,
                            TerminationCallback const& terminationCallback);

    storm::pomdp::beliefs::FirstStateNextStateGenerator<PomdpType, BeliefType> firstStateNextStateGenerator;
};
}  // namespace storm::pomdp::beliefs