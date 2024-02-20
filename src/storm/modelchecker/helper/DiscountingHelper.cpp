#include "DiscountingHelper.h"
#include "solver/helper/DiscountedValueIterationHelper.h"
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include "storm/storage/SparseMatrix.h"

namespace storm {
namespace modelchecker {
namespace helper {

template<typename ValueType>
DiscountingHelper<ValueType>::DiscountingHelper(storm::storage::SparseMatrix<ValueType> const& A) : localA(nullptr), A(&A) {
    progressMeasurement = storm::utility::ProgressMeasurement("iterations");
}

template<typename ValueType>
void DiscountingHelper<ValueType>::setUpViOperator() const {
    if (!viOperator) {
        viOperator = std::make_shared<solver::helper::ValueIterationOperator<ValueType, false>>();
        viOperator->setMatrixBackwards(*this->A);
    }
    /*if (this->choiceFixedForRowGroup) {
        // Ignore those rows that are not selected
        assert(this->initialScheduler);
        auto callback = [&](uint64_t groupIndex, uint64_t localRowIndex) {
            return this->choiceFixedForRowGroup->get(groupIndex) && this->initialScheduler->at(groupIndex) != localRowIndex;
        };
        viOperator->setIgnoredRows(true, callback);
    }*/
}

template<typename ValueType>
void DiscountingHelper<ValueType>::showProgressIterative(uint64_t iteration) const {
    progressMeasurement->updateProgress(iteration);
}

template<typename ValueType>
bool DiscountingHelper<ValueType>::solveWithDiscountedValueIteration(storm::Environment const& env, std::optional<OptimizationDirection> dir,
                                                                     std::vector<ValueType>& x, std::vector<ValueType> const& b, ValueType discountFactor) const {
    storm::solver::helper::DiscountedValueIterationHelper<ValueType, false> viHelper(viOperator);
    uint64_t numIterations{0};
    auto viCallback = [&](solver::SolverStatus const& current) {
        return current;
        showProgressIterative(numIterations);
        // return this->updateStatus(current, x, guarantee, numIterations, env.solver().minMax().getMaximalNumberOfIterations());
    };
    progressMeasurement->startNewMeasurement(0);
    auto status = viHelper.DiscountedVI(x, b, numIterations, env.solver().minMax().getRelativeTerminationCriterion(),
                                        storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision()), discountFactor, dir, viCallback,
                                        env.solver().minMax().getMultiplicationStyle());

    // this->reportStatus(status, numIterations);
    return status == solver::SolverStatus::Converged || status == solver::SolverStatus::TerminatedEarly;
}

template class DiscountingHelper<double>;
template class DiscountingHelper<storm::RationalNumber>;
}  // namespace helper
}  // namespace modelchecker
}  // namespace storm
