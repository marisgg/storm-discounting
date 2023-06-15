#include "GoalHsviModelChecker.h"

#include "storm-pomdp/analysis/FormulaInformation.h"
#include "storm-pomdp/modelchecker/PreprocessingPomdpValueBoundsModelChecker.h"
#include "storm/models/sparse/Pomdp.h"

namespace storm {
namespace pomdp {
namespace modelchecker {

template<typename PomdpModelType, typename BeliefValueType>
GoalHsviModelChecker<PomdpModelType, BeliefValueType>::Result::Result(ValueType lower, ValueType upper) : lowerBound(lower), upperBound(upper) {
    // Intentionally left empty
}

template<typename PomdpModelType, typename BeliefValueType>
GoalHsviModelChecker<PomdpModelType, BeliefValueType>::GoalHsviModelChecker(std::shared_ptr<PomdpModelType> pomdp) : inputPomdp(pomdp) {
    // Intentionally left empty
}

template<typename PomdpModelType, typename BeliefValueType>
typename GoalHsviModelChecker<PomdpModelType, BeliefValueType>::Result GoalHsviModelChecker<PomdpModelType, BeliefValueType>::check(
    storm::logic::Formula const& formula) {
    // For now, we assume the setting of the Goal HSVI paper where we minimise positive costs (rewards)
    storm::pomdp::analysis::FormulaInformation formulaInfo = storm::pomdp::analysis::getFormulaInformation(*inputPomdp, formula);
    STORM_LOG_ASSERT(formulaInfo.isNonNestedExpectedRewardFormula() && formulaInfo.minimize(),
                     "Goal HSVI does not support formulas other than minimisation of positive rewards.");
    auto preProcessingMC = PreprocessingPomdpValueBoundsModelChecker<ValueType>(*inputPomdp);
    std::vector<ValueType> uniformBound = preProcessingMC.computeValuesForUniformPolicy(formula, formulaInfo).first;

    auto lowerBound = storm::utility::zero<ValueType>();
    ValueType upperBound = uniformBound[inputPomdp->getInitialStates().getNextSetIndex(0)];

    // ADD THE CODE FOR GOAL HSVI HERE

    return Result(lowerBound, upperBound);
}

/* Template Instantiations */

template class GoalHsviModelChecker<storm::models::sparse::Pomdp<double>>;

template class GoalHsviModelChecker<storm::models::sparse::Pomdp<double>, storm::RationalNumber>;

template class GoalHsviModelChecker<storm::models::sparse::Pomdp<storm::RationalNumber>, double>;

template class GoalHsviModelChecker<storm::models::sparse::Pomdp<storm::RationalNumber>>;

}  // namespace modelchecker
}  // namespace pomdp
}  // namespace storm