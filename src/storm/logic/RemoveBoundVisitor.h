#pragma once

#include "storm/logic/CloneVisitor.h"
#include "storm/storage/expressions/Expression.h"

namespace storm::logic {

class RemoveBoundVisitor : public CloneVisitor {
   public:
    RemoveBoundVisitor() = default;

    /*!
     * Removes bounds of bounded Untils, leaves everything else as is
     * @param f The formula in which bounds are to be removed
     * @return A formula that is the same as the given formula, except that it has no bounds
     */
    std::shared_ptr<Formula> dropBounds(Formula const& f) const;

    virtual boost::any visit(BoundedUntilFormula const& f, boost::any const& data) const override;
};

}  // namespace storm::logic