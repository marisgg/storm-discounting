#include "storm/solver/helper/DiscountedValueIterationHelper.h"

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/solver/helper/ValueIterationOperator.h"
#include "storm/utility/Extremum.h"

namespace storm::solver::helper {

template<typename ValueType, storm::OptimizationDirection Dir, bool Relative>
class DiscountedVIOperatorBackend {
   public:
    DiscountedVIOperatorBackend(ValueType const& precision, ValueType const& discountFactor)
        : precision{precision},
          discountFactor{discountFactor},
          bound{(((storm::utility::one<ValueType>() - discountFactor) * precision) / (2 * discountFactor))} {
        // Intentionally left empty
    }

    void startNewIteration() {
        isConverged = true;
    }

    void firstRow(ValueType&& value, [[maybe_unused]] uint64_t rowGroup, [[maybe_unused]] uint64_t row) {
        best = std::move(value);
    }

    void nextRow(ValueType&& value, [[maybe_unused]] uint64_t rowGroup, [[maybe_unused]] uint64_t row) {
        best &= value;
    }

    void applyUpdate(ValueType& currValue, [[maybe_unused]] uint64_t rowGroup) {
        if (isConverged) {
            if constexpr (Relative) {
                isConverged = storm::utility::abs<ValueType>(currValue - *best) <= storm::utility::abs<ValueType>(bound * currValue);
            } else {
                isConverged = storm::utility::abs<ValueType>(currValue - *best) <= bound;
            }
        }
        currValue = std::move(*best);
    }

    void endOfIteration() const {
        // intentionally left empty.
    }

    bool converged() const {
        return isConverged;
    }

    bool constexpr abort() const {
        return false;
    }

   private:
    storm::utility::Extremum<Dir, ValueType> best;
    ValueType const precision;
    ValueType const discountFactor;
    ValueType const bound;
    bool isConverged{true};
};

template<typename ValueType, bool TrivialRowGrouping>
DiscountedValueIterationHelper<ValueType, TrivialRowGrouping>::DiscountedValueIterationHelper(
    std::shared_ptr<ValueIterationOperator<ValueType, TrivialRowGrouping>> viOperator)
    : viOperator(viOperator) {
    // Intentionally left empty
}

template<typename ValueType, bool TrivialRowGrouping>
template<storm::OptimizationDirection Dir, bool Relative>
SolverStatus DiscountedValueIterationHelper<ValueType, TrivialRowGrouping>::DiscountedVI(
    std::vector<ValueType>& operand, std::vector<ValueType> const& offsets, uint64_t& numIterations, ValueType const& precision,
    ValueType const& discountFactor, std::function<SolverStatus(SolverStatus const&)> const& iterationCallback, MultiplicationStyle mult) const {
    DiscountedVIOperatorBackend<ValueType, Dir, Relative> backend{precision, discountFactor};
    std::vector<ValueType>* operand1{&operand};
    std::vector<ValueType>* operand2{&operand};
    if (mult == MultiplicationStyle::Regular) {
        operand2 = &viOperator->allocateAuxiliaryVector(operand.size());
    }
    bool resultInAuxVector{false};
    SolverStatus status{SolverStatus::InProgress};
    while (status == SolverStatus::InProgress) {
        ++numIterations;
        if (viOperator->template applyWithDiscounting(*operand1, *operand2, offsets, backend, discountFactor)) {
            status = SolverStatus::Converged;
        } else if (iterationCallback) {
            status = iterationCallback(status);
        }
        if (mult == MultiplicationStyle::Regular) {
            std::swap(operand1, operand2);
            resultInAuxVector = !resultInAuxVector;
        }
    }
    if (mult == MultiplicationStyle::Regular) {
        if (resultInAuxVector) {
            STORM_LOG_ASSERT(&operand == operand2, "Unexpected operand address");
            std::swap(*operand1, *operand2);
        }
        viOperator->freeAuxiliaryVector();
    }
    return status;
}

template<typename ValueType, bool TrivialRowGrouping>
SolverStatus DiscountedValueIterationHelper<ValueType, TrivialRowGrouping>::DiscountedVI(
    std::vector<ValueType>& operand, std::vector<ValueType> const& offsets, uint64_t& numIterations, bool relative, ValueType const& precision,
    ValueType const& discountFactor, std::optional<storm::OptimizationDirection> const& dir,
    std::function<SolverStatus(SolverStatus const&)> const& iterationCallback, MultiplicationStyle mult) const {
    STORM_LOG_ASSERT(TrivialRowGrouping || dir.has_value(), "no optimization direction given!");
    if (!dir.has_value() || maximize(*dir)) {
        if (relative) {
            return DiscountedVI<storm::OptimizationDirection::Maximize, true>(operand, offsets, numIterations, precision, discountFactor, iterationCallback,
                                                                              mult);
        } else {
            return DiscountedVI<storm::OptimizationDirection::Maximize, false>(operand, offsets, numIterations, precision, discountFactor, iterationCallback,
                                                                               mult);
        }
    } else {
        if (relative) {
            return DiscountedVI<storm::OptimizationDirection::Minimize, true>(operand, offsets, numIterations, precision, discountFactor, iterationCallback,
                                                                              mult);
        } else {
            return DiscountedVI<storm::OptimizationDirection::Minimize, false>(operand, offsets, numIterations, precision, discountFactor, iterationCallback,
                                                                               mult);
        }
    }
}

template<typename ValueType, bool TrivialRowGrouping>
SolverStatus DiscountedValueIterationHelper<ValueType, TrivialRowGrouping>::DiscountedVI(
    std::vector<ValueType>& operand, std::vector<ValueType> const& offsets, bool relative, ValueType const& precision, ValueType const& discountFactor,
    std::optional<storm::OptimizationDirection> const& dir, std::function<SolverStatus(SolverStatus const&)> const& iterationCallback,
    MultiplicationStyle mult) const {
    uint64_t numIterations = 0;
    return DiscountedVI(operand, offsets, numIterations, relative, precision, discountFactor, dir, iterationCallback, mult);
}

template class DiscountedValueIterationHelper<double, true>;
template class DiscountedValueIterationHelper<double, false>;
template class DiscountedValueIterationHelper<storm::RationalNumber, true>;
template class DiscountedValueIterationHelper<storm::RationalNumber, false>;

}  // namespace storm::solver::helper
