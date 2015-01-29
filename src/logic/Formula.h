#ifndef STORM_LOGIC_FORMULA_H_
#define STORM_LOGIC_FORMULA_H_

#include <memory>
#include <iostream>

#include "src/modelchecker/results/CheckResult.h"

namespace storm {
    namespace logic {
        // Forward-declare all formula classes.
        class PathFormula;
        class StateFormula;
        class BinaryStateFormula;
        class UnaryStateFormula;
        class BinaryBooleanStateFormula;
        class UnaryBooleanStateFormula;
        class BooleanLiteralFormula;
        class AtomicExpressionFormula;
        class AtomicLabelFormula;
        class UntilFormula;
        class BoundedUntilFormula;
        class EventuallyFormula;
        class GloballyFormula;
        class BinaryPathFormula;
        class UnaryPathFormula;
        class ConditionalPathFormula;
        class NextFormula;
        class LongRunAverageOperatorFormula;
        class ExpectedTimeOperatorFormula;
        class RewardPathFormula;
        class CumulativeRewardFormula;
        class InstantaneousRewardFormula;
        class ReachabilityRewardFormula;
        class ProbabilityOperatorFormula;
        class RewardOperatorFormula;

        // Also foward-declare base model checker class.
        class ModelChecker;
        
        class Formula : public std::enable_shared_from_this<Formula const> {
        public:
            // Make the destructor virtual to allow deletion of objects of subclasses via a pointer to this class.
            virtual ~Formula() {
                // Intentionally left empty.
            };
            
            friend std::ostream& operator<<(std::ostream& out, Formula const& formula);
            
            // Methods for querying the exact formula type.
            virtual bool isPathFormula() const;
            virtual bool isStateFormula() const;
            virtual bool isBinaryStateFormula() const;
            virtual bool isUnaryStateFormula() const;
            virtual bool isBinaryBooleanStateFormula() const;
            virtual bool isUnaryBooleanStateFormula() const;
            virtual bool isBooleanLiteralFormula() const;
            virtual bool isTrueFormula() const;
            virtual bool isFalseFormula() const;
            virtual bool isAtomicExpressionFormula() const;
            virtual bool isAtomicLabelFormula() const;
            virtual bool isUntilFormula() const;
            virtual bool isBoundedUntilFormula() const;
            virtual bool isEventuallyFormula() const;
            virtual bool isGloballyFormula() const;
            virtual bool isBinaryPathFormula() const;
            virtual bool isUnaryPathFormula() const;
            virtual bool isConditionalPathFormula() const;
            virtual bool isNextFormula() const;
            virtual bool isLongRunAverageOperatorFormula() const;
            virtual bool isExpectedTimeOperatorFormula() const;
            virtual bool isRewardPathFormula() const;
            virtual bool isCumulativeRewardFormula() const;
            virtual bool isInstantaneousRewardFormula() const;
            virtual bool isReachabilityRewardFormula() const;
            virtual bool isProbabilityOperatorFormula() const;
            virtual bool isRewardOperatorFormula() const;

            virtual bool isPctlPathFormula() const;
            virtual bool isPctlStateFormula() const;
            virtual bool isCslPathFormula() const;
            virtual bool isCslStateFormula() const;
            virtual bool isPltlFormula() const;
            virtual bool isLtlFormula() const;
            virtual bool isPropositionalFormula() const;
            virtual bool containsProbabilityOperator() const;
            virtual bool containsNestedProbabilityOperators() const;
            virtual bool containsRewardOperator() const;
            virtual bool containsNestedRewardOperators() const;
            
            static std::shared_ptr<Formula const> getTrueFormula();
            
            PathFormula& asPathFormula();
            PathFormula const& asPathFormula() const;
        
            StateFormula& asStateFormula();
            StateFormula const& asStateFormula() const;
            
            BinaryStateFormula& asBinaryStateFormula();
            BinaryStateFormula const& asBinaryStateFormula() const;
            
            UnaryStateFormula& asUnaryStateFormula();
            UnaryStateFormula const& asUnaryStateFormula() const;
            
            BinaryBooleanStateFormula& asBinaryBooleanStateFormula();
            BinaryBooleanStateFormula const& asBinaryBooleanStateFormula() const;

            UnaryBooleanStateFormula& asUnaryBooleanStateFormula();
            UnaryBooleanStateFormula const& asUnaryBooleanStateFormula() const;

            BooleanLiteralFormula& asBooleanLiteralFormula();
            BooleanLiteralFormula const& asBooleanLiteralFormula() const;
            
            AtomicExpressionFormula& asAtomicExpressionFormula();
            AtomicExpressionFormula const& asAtomicExpressionFormula() const;
            
            AtomicLabelFormula& asAtomicLabelFormula();
            AtomicLabelFormula const& asAtomicLabelFormula() const;
            
            UntilFormula& asUntilFormula();
            UntilFormula const& asUntilFormula() const;
            
            BoundedUntilFormula& asBoundedUntilFormula();
            BoundedUntilFormula const& asBoundedUntilFormula() const;
            
            EventuallyFormula& asEventuallyFormula();
            EventuallyFormula const& asEventuallyFormula() const;
            
            GloballyFormula& asGloballyFormula();
            GloballyFormula const& asGloballyFormula() const;
            
            BinaryPathFormula& asBinaryPathFormula();
            BinaryPathFormula const& asBinaryPathFormula() const;
            
            UnaryPathFormula& asUnaryPathFormula();
            UnaryPathFormula const& asUnaryPathFormula() const;
            
            ConditionalPathFormula& asConditionalPathFormula();
            ConditionalPathFormula const& asConditionalPathFormula() const;
            
            NextFormula& asNextFormula();
            NextFormula const& asNextFormula() const;
            
            LongRunAverageOperatorFormula& asLongRunAverageOperatorFormula();
            LongRunAverageOperatorFormula const& asLongRunAverageOperatorFormula() const;

            ExpectedTimeOperatorFormula& asExpectedTimeOperatorFormula();
            ExpectedTimeOperatorFormula const& asExpectedTimeOperatorFormula() const;
            
            RewardPathFormula& asRewardPathFormula();
            RewardPathFormula const& asRewardPathFormula() const;
            
            CumulativeRewardFormula& asCumulativeRewardFormula();
            CumulativeRewardFormula const& asCumulativeRewardFormula() const;
            
            InstantaneousRewardFormula& asInstantaneousRewardFormula();
            InstantaneousRewardFormula const& asInstantaneousRewardFormula() const;
            
            ReachabilityRewardFormula& asReachabilityRewardFormula();
            ReachabilityRewardFormula const& asReachabilityRewardFormula() const;
            
            ProbabilityOperatorFormula& asProbabilityOperatorFormula();
            ProbabilityOperatorFormula const& asProbabilityOperatorFormula() const;
            
            RewardOperatorFormula& asRewardOperatorFormula();
            RewardOperatorFormula const& asRewardOperatorFormula() const;
            
            std::vector<std::shared_ptr<AtomicExpressionFormula const>> getAtomicExpressionFormulas() const;
            std::vector<std::shared_ptr<AtomicLabelFormula const>> getAtomicLabelFormulas() const;
            
            std::shared_ptr<Formula const> asSharedPointer();
            std::shared_ptr<Formula const> asSharedPointer() const;
            
            virtual std::ostream& writeToStream(std::ostream& out) const = 0;
            
            virtual void gatherAtomicExpressionFormulas(std::vector<std::shared_ptr<AtomicExpressionFormula const>>& atomicExpressionFormulas) const;
            virtual void gatherAtomicLabelFormulas(std::vector<std::shared_ptr<AtomicLabelFormula const>>& atomicExpressionFormulas) const;
            
        private:
            // Currently empty.
        };
        
        std::ostream& operator<<(std::ostream& out, Formula const& formula);
    }
}

#endif /* STORM_LOGIC_FORMULA_H_ */