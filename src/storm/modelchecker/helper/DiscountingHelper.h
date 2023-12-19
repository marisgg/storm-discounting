#ifndef STORM_MODEL_CHECKER_DISCOUNTINGHELPER_H
#define STORM_MODEL_CHECKER_DISCOUNTINGHELPER_H

#include <solver/helper/ValueIterationOperator.h>
#include "SingleValueModelCheckerHelper.h"

namespace storm {
class Environment;
namespace modelchecker {
namespace helper {
template<typename ValueType>
class DiscountingHelper : public SingleValueModelCheckerHelper<ValueType, storm::models::ModelRepresentation::Sparse> {
   public:
    DiscountingHelper();

    void setUpViOperator() const;

    bool solveWithDiscountedValueIteration(Environment const& env, OptimizationDirection dir, std::vector<ValueType>& x, std::vector<ValueType> const& b,
                                           ValueType discountFactor) const;

   private:
    mutable std::shared_ptr<storm::solver::helper::ValueIterationOperator<ValueType, false>> viOperator;
    mutable std::unique_ptr<std::vector<ValueType>> auxiliaryRowGroupVector;
};
}  // namespace helper
}  // namespace modelchecker
}  // namespace storm
#endif  // STORM_MODEL_CHECKER_DISCOUNTINGHELPER_H
