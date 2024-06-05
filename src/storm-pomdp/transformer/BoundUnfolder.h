//
// Created by spook on 26.04.23.
//

#ifndef STORM_BOUNDUNFOLDER_H
#define STORM_BOUNDUNFOLDER_H
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
        storm::logic::ProbabilityOperatorFormula formula;
        std::map<std::pair<uint_fast64_t, ValueType>, uint_fast64_t> stateEpochToNewState;
        std::map<uint_fast64_t, std::pair<uint_fast64_t, ValueType>> newStateToStateEpoch;
        UnfoldingResult(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp, storm::logic::ProbabilityOperatorFormula formula, std::map<std::pair<uint_fast64_t, ValueType>, uint_fast64_t> stateEpochToNewState, std::map<uint_fast64_t, std::pair<uint_fast64_t, ValueType>> newStateToStateEpoch)
            : pomdp(pomdp), formula(formula), stateEpochToNewState(stateEpochToNewState), newStateToStateEpoch(newStateToStateEpoch){}
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
    ValueType getBound(const storm::logic::Formula& formula);
};

}  // namespace transformer
}  // namespace storm
#endif  // STORM_BOUNDUNFOLDER_H
