#pragma once

#include <logic/TimeBoundType.h>

#include <utility>
#include "logic/ProbabilityOperatorFormula.h"
#include "logic/QuantileFormula.h"
#include "models/sparse/Pomdp.h"

namespace storm::transformer {
template<typename ValueType>
class BoundUnfolder {
   public:
    struct UnfoldingResult {
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp;
        std::shared_ptr<storm::logic::Formula> formula;
        std::vector<std::unordered_map<std::string, ValueType>> idToEpochMap;
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> stateEpochToNewState;
        std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> newStateToStateEpoch;
        UnfoldingResult(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp, std::shared_ptr<storm::logic::Formula> formula,
                        std::vector<std::unordered_map<std::string, ValueType>> idToEpochMap,
                        std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> stateEpochsToNewState,
                        std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> newStateToStateEpochs)
            : pomdp(pomdp),
              formula(std::move(formula)),
              idToEpochMap(std::move(idToEpochMap)),
              stateEpochToNewState(std::move(stateEpochsToNewState)),
              newStateToStateEpoch(std::move(newStateToStateEpochs)) {}
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

}  // namespace storm::transformer
