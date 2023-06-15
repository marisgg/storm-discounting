#pragma once

#include "storm-pomdp/storage/BeliefManager.h"

namespace storm {
namespace models {
namespace sparse {
template<class ValueType, typename RewardModelType>
class Pomdp;
}
}  // namespace models
namespace logic {
class Formula;
}
namespace pomdp {
namespace modelchecker {

template<typename PomdpModelType, typename BeliefValueType = typename PomdpModelType::ValueType>
class GoalHsviModelChecker {
    typedef typename PomdpModelType::ValueType ValueType;

    struct Result {
        Result(ValueType lower, ValueType upper);
        ValueType lowerBound;
        ValueType upperBound;
    };

   public:
    /**
     * Constructor
     * @param pomdp pointer to the POMDP to be checked
     */
    explicit GoalHsviModelChecker(std::shared_ptr<PomdpModelType> pomdp);

    /**
     * Performs model checking of the given POMDP with regards to a formula using Goal HSVI
     * @param formula the formula to check
     * // ADD THE PARAMETERS YOU NEED IN THE CHECK PROCEDURE HERE
     */
    Result check(storm::logic::Formula const& formula);

    // Add additional public functions here

   private:
    // Add additional private functions here

    std::shared_ptr<PomdpModelType> inputPomdp;
};

}  // namespace modelchecker
}  // namespace pomdp
}  // namespace storm