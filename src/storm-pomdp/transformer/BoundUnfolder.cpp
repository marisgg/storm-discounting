//
// Created by spook on 26.04.23.
//

#include "BoundUnfolder.h"

namespace storm {
    namespace transformer {
        template<typename ValueType>
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> BoundUnfolder<ValueType>::unfold(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPOMDP, const storm::logic::QuantileFormula& formula) {
            return std::shared_ptr<storm::models::sparse::Pomdp<ValueType>>();
        }
    }
}