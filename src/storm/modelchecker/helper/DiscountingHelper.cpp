#include "DiscountingHelper.h"
#include "exceptions/IllegalFunctionCallException.h"
#include "solver/helper/DiscountedValueIterationHelper.h"
#include "solver/helper/SchedulerTrackingHelper.h"
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
DiscountingHelper<ValueType>::DiscountingHelper(storm::storage::SparseMatrix<ValueType> const& A, bool trackScheduler)
    : localA(nullptr), A(&A), trackScheduler(trackScheduler) {
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
                                                                     std::vector<ValueType>& x, std::vector<ValueType> const& b,
                                                                     ValueType discountFactor) const {
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

    // If requested, we store the scheduler for retrieval.
    if (this->isTrackSchedulerSet()) {
        this->extractScheduler(x, b, dir.value(), true);
    }
    return status == solver::SolverStatus::Converged || status == solver::SolverStatus::TerminatedEarly;
}

template<typename ValueType>
storm::storage::Scheduler<ValueType> DiscountingHelper<ValueType>::computeScheduler() const {
    STORM_LOG_THROW(hasScheduler(), storm::exceptions::IllegalFunctionCallException, "Cannot retrieve scheduler, because none was generated.");
    storm::storage::Scheduler<ValueType> result(schedulerChoices->size());
    uint_fast64_t state = 0;
    for (auto const& schedulerChoice : schedulerChoices.get()) {
        result.setChoice(schedulerChoice, state);
        ++state;
    }
    return result;
}

template<typename ValueType>
bool DiscountingHelper<ValueType>::hasScheduler() const {
    return static_cast<bool>(schedulerChoices);
}

template<typename ValueType>
void DiscountingHelper<ValueType>::extractScheduler(std::vector<ValueType>& x, std::vector<ValueType> const& b, OptimizationDirection const& dir,
                                                    bool robust) const {
    // Make sure that storage for scheduler choices is available
    if (!this->schedulerChoices) {
        this->schedulerChoices = std::vector<uint64_t>(x.size(), 0);
    } else {
        this->schedulerChoices->resize(x.size(), 0);
    }
    // Set the correct choices.
    STORM_LOG_WARN_COND(viOperator, "Expected VI operator to be initialized for scheduler extraction. Initializing now, but this is inefficient.");
    if (!viOperator) {
        setUpViOperator();
    }
    storm::solver::helper::SchedulerTrackingHelper<ValueType> schedHelper(viOperator);
    schedHelper.computeScheduler(x, b, dir, *this->schedulerChoices, robust, nullptr);
}

template<typename ValueType>
void DiscountingHelper<ValueType>::setTrackScheduler(bool trackScheduler) {
    this->trackScheduler = trackScheduler;
    if (!this->trackScheduler) {
        schedulerChoices = boost::none;
    }
}

template<typename ValueType>
bool DiscountingHelper<ValueType>::isTrackSchedulerSet() const {
    return this->trackScheduler;
}

template class DiscountingHelper<double>;
template class DiscountingHelper<storm::RationalNumber>;
}  // namespace helper
}  // namespace modelchecker
}  // namespace storm
