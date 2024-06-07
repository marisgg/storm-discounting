#include "BoundToUnboundVisitor.h"

#include <boost/any.hpp>

#include "storm/logic/Formulas.h"

namespace storm {
namespace logic {

std::shared_ptr<Formula> BoundToUnboundVisitor::dropBounds(const storm::logic::Formula& f) const {
    boost::any result = f.accept(*this, boost::any());
    return boost::any_cast<std::shared_ptr<Formula>>(result);
}

boost::any BoundToUnboundVisitor::visit(BoundedUntilFormula const& f, boost::any const&) const {
    STORM_LOG_ASSERT(!f.hasMultiDimensionalSubformulas(), "Cannot turn a bounded Until formula into an unbounded one if it has multidimensional subformulas!");
    auto left = f.getLeftSubformula().clone();
    auto right = f.getRightSubformula().clone();
    return std::static_pointer_cast<Formula>(std::make_shared<UntilFormula>(left, right));
}

}  // namespace logic
}  // namespace storm