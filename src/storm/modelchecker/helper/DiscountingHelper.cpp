#include "DiscountingHelper.h"
#include "solver/helper/DiscountedValueIterationHelper.h"
#include "storm/environment/solver/MinMaxSolverEnvironment.h"

namespace storm {
namespace modelchecker {
namespace helper {

template<typename ValueType>
void DiscountingHelper<ValueType>::setUpViOperator() const {
    if (!viOperator) {
        viOperator = std::make_shared<solver::helper::ValueIterationOperator<ValueType, false>>();
        viOperator->setMatrixBackwards(*this->A);
    }
    if (this->choiceFixedForRowGroup) {
        // Ignore those rows that are not selected
        assert(this->initialScheduler);
        auto callback = [&](uint64_t groupIndex, uint64_t localRowIndex) {
            return this->choiceFixedForRowGroup->get(groupIndex) && this->initialScheduler->at(groupIndex) != localRowIndex;
        };
        viOperator->setIgnoredRows(true, callback);
    }
}

template<typename ValueType>
bool DiscountingHelper<ValueType>::solveWithDiscountedValueIteration(storm::Environment const& env, OptimizationDirection dir, std::vector<ValueType>& x,
                                                                     std::vector<ValueType> const& b, ValueType discountFactor) const {
    storm::solver::helper::DiscountedValueIterationHelper<ValueType, false> viHelper(viOperator);
    uint64_t numIterations{0};
    auto viCallback = [&](solver::SolverStatus const& current) {
        /*%this->showProgressIterative(numIterations);
        return this->updateStatus(current, x, guarantee, numIterations, env.solver().minMax().getMaximalNumberOfIterations());*/
    };
    // this->startMeasureProgress();
    auto status = viHelper.DiscountedVI(x, b, numIterations, env.solver().minMax().getRelativeTerminationCriterion(),
                                        storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision()), discountFactor, dir, viCallback,
                                        env.solver().minMax().getMultiplicationStyle());

    // this->reportStatus(status, numIterations);
}
}  // namespace helper
}  // namespace modelchecker
}  // namespace storm
