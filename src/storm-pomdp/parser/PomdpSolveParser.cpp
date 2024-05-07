#include "PomdpSolveParser.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/api/builder.h"

namespace storm {
namespace pomdp {
namespace parser {

template<typename ValueType>
PomdpSolveParserResult<ValueType> PomdpSolveParser<ValueType>::parsePomdpSolveFile(std::string const& filename) {
    STORM_LOG_WARN("POMDPsolve parser not implemented yet.");
    // Implement parsing here.
    // TODO read input
    // TODO prepare data structures
    // TODO build POMDP from data structures
    return {};
}

template class PomdpSolveParser<double>;
template class PomdpSolveParser<storm::RationalNumber>;
}  // namespace parser
}  // namespace pomdp
}  // namespace storm
