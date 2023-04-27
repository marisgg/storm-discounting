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
            std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> unfold(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPOMDP, const storm::logic::QuantileFormula& formula);
           private:
        };

        }
}
#endif  // STORM_BOUNDUNFOLDER_H
