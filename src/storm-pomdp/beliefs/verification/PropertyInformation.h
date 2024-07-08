#pragma once

#include <optional>
#include <set>
#include <string>

#include "storm-pomdp/beliefs/utility/types.h"
#include "storm/logic/TimeBound.h"
#include "storm/solver/OptimizationDirection.h"

namespace storm::pomdp::beliefs {
struct RewardBound {
    std::string rewardModelName;
    std::optional<storm::logic::TimeBound> lowerBound;
    std::optional<storm::logic::TimeBound> upperBound;
};

struct PropertyInformation {
    enum class Kind { ReachabilityProbability, ExpectedTotalReachabilityReward, RewardBoundedReachabilityProbability };
    Kind kind;
    std::set<BeliefObservationType> targetObservations;
    std::optional<std::string> rewardModelName;
    storm::OptimizationDirection dir;
    std::vector<RewardBound> rewardBounds;
};
}  // namespace storm::pomdp::beliefs