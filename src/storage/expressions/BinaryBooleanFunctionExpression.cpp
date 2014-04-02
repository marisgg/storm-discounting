#include "src/storage/expressions/BinaryBooleanFunctionExpression.h"

namespace storm {
    namespace expressions {
        BinaryBooleanFunctionExpression::BinaryBooleanFunctionExpression(ExpressionReturnType returnType, std::unique_ptr<BaseExpression>&& firstOperand, std::unique_ptr<BaseExpression>&& secondOperand, OperatorType operatorType) : BinaryExpression(returnType, std::move(firstOperand), std::move(secondOperand)), operatorType(operatorType) {
            // Intentionally left empty.
        }
        
        BinaryBooleanFunctionExpression::OperatorType BinaryBooleanFunctionExpression::getOperatorType() const {
            return this->operatorType;
        }
        
        BinaryBooleanFunctionExpression::BinaryBooleanFunctionExpression(BinaryBooleanFunctionExpression const& other) : BinaryExpression(other), operatorType(other.getOperatorType()) {
            // Intentionally left empty.
        }
        
        BinaryBooleanFunctionExpression& BinaryBooleanFunctionExpression::operator=(BinaryBooleanFunctionExpression const& other) {
            if (this != &other) {
                BinaryExpression::operator=(other);
                this->operatorType = other.getOperatorType();
            }
            return *this;
        }
        
        bool BinaryBooleanFunctionExpression::evaluateAsBool(Valuation const& valuation) const {
            bool firstOperandEvaluation = this->getFirstOperand()->evaluateAsBool(valuation);
            bool secondOperandEvaluation = this->getSecondOperand()->evaluateAsBool(valuation);
            
            bool result;
            switch (this->getOperatorType()) {
                case AND: result = firstOperandEvaluation && secondOperandEvaluation; break;
                case OR: result = firstOperandEvaluation || secondOperandEvaluation; break;
            }
            
            return result;
        }
        
        std::unique_ptr<BaseExpression> BinaryBooleanFunctionExpression::simplify() const {
            std::unique_ptr<BaseExpression> firstOperandSimplified = this->getFirstOperand()->simplify();
            std::unique_ptr<BaseExpression> secondOperandSimplified = this->getSecondOperand()->simplify();
            
            switch (this->getOperatorType()) {
                case AND: if (firstOperandSimplified->isTrue()) {
                    return secondOperandSimplified;
                } else if (firstOperandSimplified->isFalse()) {
                    return firstOperandSimplified;
                } else if (secondOperandSimplified->isTrue()) {
                    return firstOperandSimplified;
                } else if (secondOperandSimplified->isFalse()) {
                    return secondOperandSimplified;
                }
                break;
                case OR: if (firstOperandSimplified->isTrue()) {
                    return firstOperandSimplified;
                } else if (firstOperandSimplified->isFalse()) {
                    return secondOperandSimplified;
                } else if (secondOperandSimplified->isTrue()) {
                    return secondOperandSimplified;
                } else if (secondOperandSimplified->isFalse()) {
                    return firstOperandSimplified;
                }
            }
            
            return std::unique_ptr<BaseExpression>(new BinaryBooleanFunctionExpression(this->getReturnType(), std::move(firstOperandSimplified), std::move(secondOperandSimplified), this->getOperatorType()));
        }
        
        void BinaryBooleanFunctionExpression::accept(ExpressionVisitor* visitor) const {
            visitor->visit(this);
        }
        
        virtual std::unique_ptr<BaseExpression> clone() const override;
    }
}