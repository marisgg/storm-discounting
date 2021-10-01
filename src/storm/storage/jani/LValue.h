#pragma once

#include "storm/storage/jani/Variable.h"
#include "storm/storage/expressions/Expressions.h"

namespace storm {
    namespace jani {
        
        class LValue {
        public:
            explicit LValue(storm::jani::Variable const& variable);
            LValue(storm::jani::Variable const&, std::vector<storm::expressions::Expression> const index, std::vector<std::size_t> const& sizes);
            LValue(storm::jani::Variable const&, std::vector<std::size_t> const & sizes);
            LValue(storm::jani::Variable const&, storm::expressions::Expression const& expression, std::size_t size);

            LValue(LValue const&) = default;
            bool operator==(LValue const& other) const;

            // To check if the LValue is a variable or an array
            bool isVariable() const;
            bool isArray() const;
            storm::jani::Variable const& getVariable() const;

            // To check if (a part) of the array is accessed, so arrayIndex and arrayIndexVector are initialized
            bool isArrayAccess() const;
            // To check if the array is fully accessed, so result will be of the last child Type of the array. (bool[][] will be bool)
            bool isFullArrayAccess() const;
            std::vector<storm::expressions::Expression>& getArrayIndexVector();
            std::vector<storm::expressions::Expression> const& getArrayIndexVector() const;
            storm::expressions::Expression getArrayIndex() const;
            std::vector<std::size_t> const& getSizes() const;
            std::size_t const& getSizeAt(uint_fast64_t i) const;
            std::size_t getTotalSize() const;
            std::string getName() const;

            bool arrayIndexContainsVariable() const;
            void setArrayIndex(std::vector<storm::expressions::Expression> const& newIndex);
            
            /*!
             * Adds an array access index. Assumes that the underlying variable is an array variable that isn't fully accessed yet.
             * For example (using array variable a) a will become a[index] and a[1] will become a[1][index].
             */
            void addArrayAccessIndex(storm::expressions::Expression const& index);
            
            bool isTransient() const;
            bool operator< (LValue const& other) const;
            
            LValue changeAssignmentVariables(std::map<Variable const*, std::reference_wrapper<Variable const>> const& remapping) const;
            
            friend std::ostream& operator<<(std::ostream& stream, LValue const& lvalue);



        private:
            
            // The variable being assigned, this can either be the array variable or a variable of a different type
            storm::jani::Variable const* variable;

            // In case of an array access LValue, this is the accessed index of the array (if existing)
            boost::optional<std::vector<storm::expressions::Expression>> arrayIndexVector;
            std::vector<std::size_t> sizes;
        };
    }
}