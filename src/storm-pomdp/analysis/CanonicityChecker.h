#pragma once

#include "storm/models/sparse/Pomdp.h"
namespace storm {
namespace pomdp {
namespace analysis {
template<typename ValueType>
class CanonicityChecker {
   public:
    CanonicityChecker();

    /*!
     * Checks if a given POMDP is canonic
     * @param pomdp The POMDP to consider
     * @return true iff the POMDP is canonic
     */
    bool check(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp);

   private:
    bool checkWithLabels(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp);
    bool checkWithoutLabels(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp);
};
}  // namespace analysis
}  // namespace pomdp
}  // namespace storm
