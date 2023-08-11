#pragma once

#include "storm/storage/SparseMatrix.h"

namespace storm {
namespace pomdp {
namespace storage {
/**
 * TODO
 */
template<typename ValueType>
struct AlphaVectorPolicy {
    storm::storage::SparseMatrix<ValueType> alphaVectors;
    std::vector<std::string> actions;
};
}  // namespace storage
namespace parser {
template<typename ValueType>
class AlphaVectorPolicyParser {
   public:
    static storm::pomdp::storage::AlphaVectorPolicy<ValueType> parseAlphaVectorPolicy(std::string const& filename);

   private:
    static storm::pomdp::storage::AlphaVectorPolicy<ValueType> parse(std::string const& filename);
};
}  // namespace parser
}  // namespace pomdp
}  // namespace storm
