#pragma once

namespace storm::pomdp::beliefs {
enum explorationTerminationCriterion { MAX_EXPLORATION_SIZE, MAX_EXPLORATION_TIME, MAX_EXPLORATION_SIZE_AND_TIME, NONE };

template<typename ValueType>
struct BeliefBasedModelCheckerOptions {
    bool implicitCutOffs = false;

    // Termination criteria
    std::optional<uint64_t> maxExplorationSize = std::nullopt;
    std::optional<uint64_t> maxExplorationTime = std::nullopt;
    std::optional<uint64_t> maxExplorationDepth = std::nullopt;  // currently unused
    std::optional<ValueType> maxGapToCut = std::nullopt;

    /**
     * Get the termination criterion for the exploration
     * @return the termination criterion
     */
    [[nodiscard]] explorationTerminationCriterion getTerminationCriterion() const {
        if (maxExplorationSize.has_value() && maxExplorationTime.has_value()) {
            return explorationTerminationCriterion::MAX_EXPLORATION_SIZE_AND_TIME;
        } else if (maxExplorationSize.has_value()) {
            return explorationTerminationCriterion::MAX_EXPLORATION_SIZE;
        } else if (maxExplorationTime.has_value()) {
            return explorationTerminationCriterion::MAX_EXPLORATION_TIME;
        } else {
            return explorationTerminationCriterion::NONE;
        }
    }
};
}  // namespace storm::pomdp::beliefs
