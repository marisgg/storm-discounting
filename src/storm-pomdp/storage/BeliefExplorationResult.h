#pragma once

namespace storm::storage {
template<typename ValueType>
class Scheduler;
}
namespace storm::pomdp::storage {
/**
 * Struct used to store the results of the model checker
 */
template<typename ValueType>
struct BeliefExplorationResult {
    BeliefExplorationResult(ValueType lower, ValueType upper) : lowerBound(lower), upperBound(upper){};
    ValueType diff(bool relative = false) const {
        ValueType diff = upperBound - lowerBound;
        if (diff < storm::utility::zero<ValueType>()) {
            STORM_LOG_WARN_COND(diff >= storm::utility::convertNumber<ValueType>(1e-6),
                                "Upper bound '" << upperBound << "' is smaller than lower bound '" << lowerBound << "': Difference is " << diff << ".");
            diff = storm::utility::zero<ValueType>();
        }
        if (relative && !storm::utility::isZero(upperBound)) {
            diff /= upperBound;
        }
        return diff;
    };
    bool updateLowerBound(ValueType const& value) {
        if (value > lowerBound) {
            lowerBound = value;
            return true;
        }
        return false;
    };

    bool updateUpperBound(ValueType const& value) {
        if (value < upperBound) {
            upperBound = value;
            return true;
        }
        return false;
    };

    ValueType lowerBound;
    ValueType upperBound;
    std::shared_ptr<storm::models::sparse::Model<ValueType>> schedulerAsMarkovChain;
    std::vector<storm::storage::Scheduler<ValueType>> cutoffSchedulers;
};
}  // namespace storm::pomdp::storage
