#pragma once

#include "storm/models/sparse/Pomdp.h"
namespace storm {
    namespace pomdp {
        namespace analysis {
            class CanonicityChecker {
            public:
                bool check(storm::models::sparse::Pomdp pomdp);
            };
        }
    }
}
