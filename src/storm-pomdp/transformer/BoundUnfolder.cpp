//
// Created by spook on 26.04.23.
//

#include "BoundUnfolder.h"
#include <queue>
#include "api/properties.h"
#include "logic/UntilFormula.h"
#include "storage/jani/Property.h"
#include "storm-parsers/api/properties.h"
#include "storm-pomdp/analysis/FormulaInformation.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/logic/BoundedUntilFormula.h"

namespace storm {
namespace transformer {
template<typename ValueType>
std::pair<std::shared_ptr<storm::models::sparse::Pomdp<ValueType>>, storm::logic::ProbabilityOperatorFormula> BoundUnfolder<ValueType>::unfold(
    std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPOMDP, const storm::logic::Formula& formula) {
    // Check formula
    STORM_LOG_THROW(formula.isProbabilityOperatorFormula() && formula.asOperatorFormula().getSubformula().isBoundedUntilFormula() &&
                        formula.asOperatorFormula().getSubformula().asBoundedUntilFormula().getLeftSubformula().isTrueFormula(),
                    storm::exceptions::NotSupportedException, "Unexpected formula type of formula " << formula);

    // Grab reward model
    auto temp = std::set<std::string>();
    formula.gatherReferencedRewardModels(temp);
    assert(temp.size() == 1);
    auto rewModel = originalPOMDP->getRewardModel(*temp.begin());
    STORM_LOG_THROW(rewModel.hasStateActionRewards(), storm::exceptions::NotSupportedException, "Only state action rewards are currently supported.");

    // Grab bound
    ValueType bound = getBound(formula);

    // Grab matrix (mostly for coding convenience to just have it in a variable here)
    auto& ogMatrix = originalPOMDP->getTransitionMatrix();

    // Grab goal states
    auto formulaInfo = storm::pomdp::analysis::getFormulaInformation(*originalPOMDP, formula);
    auto targetStates = formulaInfo.getTargetStates().states;

    // Transformation information + variables (remove non-necessary ones later)
    auto stateEpochToNewState = std::map<std::pair<uint_fast64_t, ValueType>, uint_fast64_t>();
    auto newStateToStateEpoch = std::map<uint_fast64_t, std::pair<uint_fast64_t, ValueType>>();
    uint_fast64_t nextNewStateIndex = 2;
    std::queue<std::pair<uint_fast64_t, ValueType>> processingQ;  // queue does BFS, if DFS is desired, change to stack

    // Information for unfolded model
    auto transitions = std::vector<std::vector<std::map<uint_fast64_t, ValueType>>>();
    // first index (vec): origin state, second index(vec): action, third index(map): destination state, value(map):probability
    auto observations = std::vector<uint32_t>();
    uint_fast64_t entryCount = 0;
    uint_fast64_t choiceCount = 0;

    // Special sink states: 0 is =), 1 is =(
    transitions.push_back(std::vector<std::map<uint_fast64_t, ValueType>>());
    transitions[0].push_back(std::map<uint_fast64_t, ValueType>());
    transitions[0][0][0] = storm::utility::one<ValueType>();
    entryCount++;
    choiceCount++;
    transitions.push_back(std::vector<std::map<uint_fast64_t, ValueType>>());
    transitions[1].push_back(std::map<uint_fast64_t, ValueType>());
    transitions[1][0][1] = storm::utility::one<ValueType>();
    entryCount++;
    choiceCount++;

    // Create init state of unfolded model
    assert(originalPOMDP->getInitialStates().getNumberOfSetBits() == 1);
    uint_fast64_t initState = originalPOMDP->getInitialStates().getNextSetIndex(0);
    auto initEpochState = std::make_pair(initState, bound);
    processingQ.push(initEpochState);
    auto numberOfActions = ogMatrix.getRowGroupSize(initState);
    transitions.push_back(std::vector<std::map<uint_fast64_t, ValueType>>());
    for (auto i = 0; i < numberOfActions; i++) {
        transitions[nextNewStateIndex].push_back(std::map<uint_fast64_t, ValueType>());
        choiceCount++;
    }
    stateEpochToNewState[initEpochState] = nextNewStateIndex;
    newStateToStateEpoch[nextNewStateIndex] = initEpochState;
    nextNewStateIndex++;

    while (!processingQ.empty()) {
        std::pair<uint_fast64_t, ValueType> currentEpochState = processingQ.front();
        processingQ.pop();
        uint_fast64_t rowGroupStart = ogMatrix.getRowGroupIndices()[currentEpochState.first];
        uint_fast64_t rowGroupSize = ogMatrix.getRowGroupSize(currentEpochState.first);
        for (auto actionIndex = 0; actionIndex < rowGroupSize; actionIndex++) {
            auto row = rowGroupStart + actionIndex;
            auto reward = rewModel.getStateActionReward(row);
            for (auto entry : ogMatrix.getRow(row)) {
                // Get successor state
                uint_fast64_t oldSuccState = entry.getColumn();
                if (currentEpochState.second >= reward) {
                    // Successor with epoch != bottom
                    if (targetStates[oldSuccState]) {
                        // Successor is goal state with epoch != bottom: Transition to =)
                        transitions[stateEpochToNewState[currentEpochState]][actionIndex][0] = entry.getValue();
                        entryCount++;
                    } else {
                        // Successor with epoch != bottom but not a goal state
                        ValueType epoch = currentEpochState.second - rewModel.getStateActionReward(row);
                        auto stateEpochSucc = std::make_pair(oldSuccState, epoch);
                        if (stateEpochToNewState.find(stateEpochSucc) == stateEpochToNewState.end()) {
                            // Unfolded successor does not exist yet: create it + add it to processing queue
                            stateEpochToNewState[stateEpochSucc] = nextNewStateIndex;
                            newStateToStateEpoch[nextNewStateIndex] = stateEpochSucc;
                            numberOfActions = ogMatrix.getRowGroupSize(oldSuccState);
                            transitions.push_back(std::vector<std::map<uint_fast64_t, ValueType>>());
                            for (auto i = 0; i < numberOfActions; i++) {
                                transitions[nextNewStateIndex].push_back(std::map<uint_fast64_t, ValueType>());
                                choiceCount++;
                            }
                            nextNewStateIndex++;
                            processingQ.push(stateEpochSucc);
                        }
                        // Add transition
                        uint_fast64_t newSuccState = stateEpochToNewState[stateEpochSucc];
                        transitions[stateEpochToNewState[currentEpochState]][actionIndex][newSuccState] = entry.getValue();
                        entryCount++;
                    }
                } else {
                    // Successor with epoch == bottom: Transition to =(
                    // TODO add case of non-goal sink states here?
                    transitions[stateEpochToNewState[currentEpochState]][actionIndex][1] = entry.getValue();
                    entryCount++;
                }
            }
        }
    }

    // Observations
    observations.push_back(originalPOMDP->getNrObservations());      // =)
    observations.push_back(originalPOMDP->getNrObservations() + 1);  // =(
    for (uint_fast64_t i = 2; i < nextNewStateIndex; i++) {
        observations.push_back(originalPOMDP->getObservation(newStateToStateEpoch[i].first));
    }

    // State labeling: single label for =)
    auto stateLabeling = storm::models::sparse::StateLabeling(stateEpochToNewState.size() + 2);
    auto labeling = storm::storage::BitVector(nextNewStateIndex, false);
    labeling.set(0);
    stateLabeling.addLabel("goal", labeling);
    labeling = storm::storage::BitVector(nextNewStateIndex, false);
    labeling.set(2);
    stateLabeling.addLabel("init", labeling);

    // Build Matrix (taken from beliefmdpexplorer + adapted)
    storm::storage::SparseMatrixBuilder<ValueType> builder(choiceCount, nextNewStateIndex, entryCount, true, true, nextNewStateIndex);
    uint_fast64_t nextMatrixRow = 0;
    for (uint_fast64_t state = 0; state < transitions.size(); state++) {
        builder.newRowGroup(nextMatrixRow);
        for (auto action = 0; action < transitions[state].size(); action++) {
            for (auto const& entry : transitions[state][action]) {
                builder.addNextValue(nextMatrixRow, entry.first, entry.second);
            }
            nextMatrixRow++;
        }
    }
    auto unfoldedTransitionMatrix = builder.build();

    // Build components
    auto components = storm::storage::sparse::ModelComponents(unfoldedTransitionMatrix, std::move(stateLabeling));
    components.observabilityClasses = observations;

    // Optional copy of choice labels
    if (originalPOMDP->hasChoiceLabeling()){
        auto newChoiceLabeling = storm::models::sparse::ChoiceLabeling(choiceCount);
        auto oldChoiceLabeling = originalPOMDP->getChoiceLabeling();
        auto newRowGroupIndices = unfoldedTransitionMatrix.getRowGroupIndices();
        auto oldRowGroupIndices = originalPOMDP->getTransitionMatrix().getRowGroupIndices();

        //assert (unfoldedTransitionMatrix.getRowGroupSize(0) + unfoldedTransitionMatrix.getRowGroupSize(1) == newRowGroupIndices[2]);

        for (uint_fast64_t newState = 2; newState < transitions.size(); newState++) {
            auto oldState = newStateToStateEpoch[newState].first;
            auto oldChoiceIndex = oldRowGroupIndices[oldState];
            auto newChoiceIndex = newRowGroupIndices[newState];
            for (auto action = 0; action < transitions[newState].size(); action++) {
                for (auto label : oldChoiceLabeling.getLabelsOfChoice(oldChoiceIndex + action)) {
                    if (!newChoiceLabeling.containsLabel(label)) {
                        newChoiceLabeling.addLabel(label);
                    }
                    newChoiceLabeling.addLabelToChoice(label, newChoiceIndex + action);
                }
            }
        }
        components.choiceLabeling = std::move(newChoiceLabeling);
    }

    // Build pomdp
    auto unfoldedPomdp = storm::models::sparse::Pomdp<ValueType>(std::move(components));
    unfoldedPomdp.setIsCanonic();


    // Generate new UntilFormula
    std::string propertyString = "Pmax=? [F\"goal\"]";
    std::vector<storm::jani::Property> propertyVector = storm::api::parseProperties(propertyString);
    storm::logic::ProbabilityOperatorFormula newFormula = storm::api::extractFormulasFromProperties(propertyVector).front()->asProbabilityOperatorFormula();

    return std::make_pair(std::make_shared<storm::models::sparse::Pomdp<ValueType>>(std::move(unfoldedPomdp)), newFormula);
}

template<>
double BoundUnfolder<double>::getBound(const storm::logic::Formula& formula) {
    STORM_LOG_THROW(formula.asOperatorFormula().getSubformula().asBoundedUntilFormula().getUpperBound().hasNumericalType(), storm::exceptions::NotSupportedException,
                    "ValueType of model and Bound ValueType not matching");
    return formula.asOperatorFormula().getSubformula().asBoundedUntilFormula().getUpperBound().evaluateAsDouble();
}

template<>
storm::RationalNumber BoundUnfolder<storm::RationalNumber>::getBound(const storm::logic::Formula& formula) {
    STORM_LOG_THROW(formula.asOperatorFormula().getSubformula().asBoundedUntilFormula().getUpperBound().hasRationalType(), storm::exceptions::NotSupportedException,
                    "ValueType of model and Bound ValueType not matching");
    return formula.asOperatorFormula().getSubformula().asBoundedUntilFormula().getUpperBound().evaluateAsRational();
}

template class BoundUnfolder<double>;
template class BoundUnfolder<storm::RationalNumber>;
}  // namespace transformer
}  // namespace storm