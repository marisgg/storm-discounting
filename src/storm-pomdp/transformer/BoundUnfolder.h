//
// Created by spook on 26.04.23.
//

#ifndef STORM_BOUNDUNFOLDER_H
#define STORM_BOUNDUNFOLDER_H
#include "logic/QuantileFormula.h"
#include "models/sparse/Pomdp.h"
namespace storm {
namespace transformer {
template<typename ValueType>
class BoundUnfolder {
   public:
    BoundUnfolder() = default;
    std::pair<std::shared_ptr<storm::models::sparse::Pomdp<ValueType>>, storm::logic::ProbabilityOperatorFormula> unfold(
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPOMDP, const storm::logic::Formula& formula);

   private:
    ValueType getBound(const storm::logic::Formula& formula);
};

}  // namespace transformer
}  // namespace storm
#endif  // STORM_BOUNDUNFOLDER_H
