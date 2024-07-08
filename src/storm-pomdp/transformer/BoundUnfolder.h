#pragma once

#include <logic/TimeBoundType.h>

#include <utility>
#include "logic/QuantileFormula.h"
#include "models/sparse/Pomdp.h"
#include "logic/ProbabilityOperatorFormula.h"

namespace storm {
namespace transformer {
template<typename ValueType>
class BoundUnfolder {
   public:
    struct UnfoldingResult {
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp;
        std::shared_ptr<storm::logic::Formula> formula;
        std::map<std::pair<uint_fast64_t, std::map<std::string, ValueType>>, uint_fast64_t> stateEpochsToNewState;
        std::map<uint_fast64_t, std::pair<uint_fast64_t, std::map<std::string, ValueType>>> newStateToStateEpochs;
        UnfoldingResult(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp, std::shared_ptr<storm::logic::Formula> formula, std::map<std::pair<uint_fast64_t, std::map<std::string, ValueType>>, uint_fast64_t> stateEpochsToNewState, std::map<uint_fast64_t, std::pair<uint_fast64_t, std::map<std::string, ValueType>>> newStateToStateEpochs)
            : pomdp(pomdp), formula(std::move(formula)), stateEpochsToNewState(stateEpochsToNewState), newStateToStateEpochs(newStateToStateEpochs){}
    };

    BoundUnfolder() = default;

    /*!
     * Unfolds a pomdp w.r.t. a reward-bounded until formula
     * @param originalPomdp The pomdp to unfold
     * @param formula The formula for which we unfold
     * @return Result struct containing the new pomdp, the new formula and mappings between (state, epoch) pairs and states in the new pomdp
     */
    UnfoldingResult unfold(
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPomdp, const storm::logic::Formula& formula);

   private:
    ValueType getUpperBound(const storm::logic::BoundedUntilFormula& formula, uint64_t i);
    ValueType getLowerBound(const storm::logic::BoundedUntilFormula& formula, uint64_t i);
    std::pair<std::unordered_map<std::string, ValueType>, std::unordered_map<std::string, ValueType>> getBounds(const storm::logic::Formula& formula);
};

}  // namespace transformer
}  // namespace storm
