#pragma once
#ifndef STORM_LOGIC_BOUNDTOUNBOUNDVISITOR_H_
#define STORM_LOGIC_BOUNDTOUNBOUNDVISITOR_H_

#include "storm/logic/CloneVisitor.h"
#include "storm/storage/expressions/Expression.h"

namespace storm::logic {

class BoundToUnboundVisitor : public CloneVisitor {
   public:
    BoundToUnboundVisitor() = default;

    /*!
     * Removes bounds of bounded Untils, leaves everything else as is
     * @param f The formula in which bounds are to be removed
     * @return A formula that is the same as the given formula, except that it has no bounds
     */
    std::shared_ptr<Formula> dropBounds(Formula const& f) const;

    virtual boost::any visit(BoundedUntilFormula const& f, boost::any const& data) const override;
};

}  // namespace storm::logic

#endif /* STORM_LOGIC_BOUNDTOUNBOUNDVISITOR_H_ */