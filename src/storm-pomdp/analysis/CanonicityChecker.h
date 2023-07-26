#pragma once

#include "storm/models/sparse/Pomdp.h"
namespace storm {
    namespace pomdp {
        namespace analysis {
        template<typename ValueType> class CanonicityChecker {
            public:
                /*!
                 * Checks if a given POMDP is canonic
                 * @param pomdp The POMDP to consider
                 * @return true iff the POMDP is canonic
                 */
                bool check(storm::models::sparse::Pomdp<ValueType> pomdp);

            private:
                bool checkWithLabels(storm::models::sparse::Pomdp<ValueType> pomdp);
                bool checkWithoutLabels(storm::models::sparse::Pomdp<ValueType> pomdp);
            };
        }
    }
}
