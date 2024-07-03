//
// Created by spook on 26.04.23.
//

#include "BoundUnfolder.h"
#include <logic/BoundToUnboundVisitor.h>
#include <queue>
#include "api/properties.h"
#include "logic/UntilFormula.h"
#include "storage/jani/Property.h"
#include "storm-parsers/api/properties.h"
#include "storm-pomdp/analysis/FormulaInformation.h"
#include "storm/adapters/RationalFunctionAdapter.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/logic/BoundedUntilFormula.h"

namespace storm {
namespace transformer {
template<typename ValueType>
typename BoundUnfolder<ValueType>::UnfoldingResult BoundUnfolder<ValueType>::unfold(
    std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPomdp, const storm::logic::Formula& formula) {
    /*std::cout << "\nORIGINAL POMDP:\n";
    originalPomdp->writeDotToStream(std::cout);
    std::cout << "\nORIGINAL FORMULA:\n";
    formula->writeToStream(std::cout);*/

    // Check formula
    STORM_LOG_THROW(formula.isProbabilityOperatorFormula() && formula.asOperatorFormula().getSubformula().isBoundedUntilFormula() &&
                        formula.asOperatorFormula().getSubformula().asBoundedUntilFormula().getLeftSubformula().isTrueFormula(),
                    storm::exceptions::NotSupportedException, "Unexpected formula type of formula " << formula);

    // TODO check that reward models are state action rewards
    //STORM_LOG_THROW(rewModel.hasStateActionRewards(), storm::exceptions::NotSupportedException, "Only state action rewards are currently supported.");

    // Grab bounds
    std::map<std::string, ValueType> upperBounds, lowerBounds;
    std::tie(upperBounds, lowerBounds) = getBounds(formula);
    uint_fast64_t totalNumberOfBounds = upperBounds.size() + lowerBounds.size();

    // Grab matrix (mostly for coding convenience to just have it in a variable here)
    auto& ogMatrix = originalPomdp->getTransitionMatrix();

    // Grab goal states
    auto formulaInfo = storm::pomdp::analysis::getFormulaInformation(*originalPomdp, formula);
    auto targetStates = formulaInfo.getTargetStates().states;

    // Transformation information + variables (remove non-necessary ones later)
    auto stateEpochsToNewState = std::map<std::pair<uint_fast64_t, std::map<std::string, ValueType>>, uint_fast64_t>();
    auto newStateToStateEpoch = std::map<uint_fast64_t, std::pair<uint_fast64_t, std::map<std::string, ValueType>>>();
    uint_fast64_t nextNewStateIndex = 2;
    std::queue<std::pair<uint_fast64_t, std::map<std::string, ValueType>>> processingQ;  // queue does BFS, if DFS is desired, change to stack

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
    STORM_LOG_ASSERT(originalPomdp->getInitialStates().getNumberOfSetBits() == 1, "This has more than one initial state");
    uint_fast64_t initState = originalPomdp->getInitialStates().getNextSetIndex(0);
    std::map<std::string, ValueType> initEpochs;
    for (const auto& rewBound : upperBounds) {
        initEpochs[rewBound.first + "_ub"] = rewBound.second;
    }
    for (const auto& rewBound : lowerBounds) {
        initEpochs[rewBound.first + "_lb"] = rewBound.second;
    }
    auto initStateEpochs = std::make_pair(initState, initEpochs);
    processingQ.push(initStateEpochs);
    auto numberOfActions = ogMatrix.getRowGroupSize(initState);
    transitions.push_back(std::vector<std::map<uint_fast64_t, ValueType>>());
    for (auto i = 0; i < numberOfActions; i++) {
        transitions[nextNewStateIndex].push_back(std::map<uint_fast64_t, ValueType>());
        choiceCount++;
    }
    stateEpochsToNewState[initStateEpochs] = nextNewStateIndex;
    newStateToStateEpoch[nextNewStateIndex] = initStateEpochs;
    nextNewStateIndex++;

    while (!processingQ.empty()) {
        std::pair<uint_fast64_t, std::map<std::string, ValueType>> currentStateEpochsPair = processingQ.front(); // the state here is a state in the original pomdp
        processingQ.pop();
        uint_fast64_t rowGroupStart = ogMatrix.getRowGroupIndices()[currentStateEpochsPair.first];
        uint_fast64_t rowGroupSize = ogMatrix.getRowGroupSize(currentStateEpochsPair.first);
        for (auto actionIndex = 0; actionIndex < rowGroupSize; actionIndex++) {
            auto row = rowGroupStart + actionIndex;
            std::map<std::string, ValueType> epochValues;

            // Collect epoch values of upper bounds
            bool upperBoundViolated = false;
            for (auto ub : upperBounds) {
                ValueType reward = originalPomdp->getRewardModel(ub.first).getStateActionReward(row);
                if (reward > currentStateEpochsPair.second[ub.first + "_ub"]) {// 0 for upper bounds means we can still keep going as long as we dont collect any more of the reward
                    // Entire action goes to =( with prob 1
                    transitions[stateEpochsToNewState[currentStateEpochsPair]][actionIndex][1] = storm::utility::one<ValueType>();
                    entryCount++;
                    upperBoundViolated = true; // for going to next action
                    break;
                } else {
                    epochValues[ub.first + "_ub"] = currentStateEpochsPair.second[ub.first + "_ub"] - reward;
                }
            }
            if (upperBoundViolated) continue;

            // Collect epoch values of lower bounds
            uint_fast64_t fulfilledLowerBounds = 0;
            for (auto lb : lowerBounds) {
                ValueType reward = originalPomdp->getRewardModel(lb.first).getStateActionReward(row);
                if (reward >= currentStateEpochsPair.second[lb.first + "_lb"]) {// 0 for lower bounds means we have fulfilled the bound
                    epochValues[lb.first + "_lb"] = storm::utility::zero<ValueType>();
                    fulfilledLowerBounds++;
                } else {
                    epochValues[lb.first + "_lb"] = currentStateEpochsPair.second[lb.first + "_lb"] - reward;
                }
            }
            bool lowerBoundsFulfilled = fulfilledLowerBounds == lowerBounds.size();

            // Per transition
            for (auto entry : ogMatrix.getRow(row)) {
                // get successor in og pomdp
                uint_fast64_t oldSuccState = entry.getColumn();
                if (targetStates[oldSuccState] && lowerBoundsFulfilled) {
                    // transition to =) with prob. of the successor (but check if there already is a transition going to =) and if so, just add to it)
                    if (transitions[stateEpochsToNewState[currentStateEpochsPair]][actionIndex].find(0) == transitions[stateEpochsToNewState[currentStateEpochsPair]][actionIndex].end()){
                        transitions[stateEpochsToNewState[currentStateEpochsPair]][actionIndex][0] = entry.getValue();
                        entryCount++;
                    } else {
                        transitions[stateEpochsToNewState[currentStateEpochsPair]][actionIndex][0] += entry.getValue();
                    }
                } else {
                    // see if successor state in unfolding exists already
                    // if not, create it + add it to the queue
                    auto succEpochs = std::make_pair(oldSuccState, epochValues);
                    auto searchRes = stateEpochsToNewState.find(succEpochs);
                    uint_fast64_t unfSuccState;
                    if (searchRes != stateEpochsToNewState.end()) { // exists already
                        unfSuccState = searchRes->second;
                    } else {
                        unfSuccState = nextNewStateIndex;
                        stateEpochsToNewState[succEpochs] = nextNewStateIndex;
                        newStateToStateEpoch[nextNewStateIndex] = succEpochs;
                        numberOfActions = ogMatrix.getRowGroupSize(oldSuccState);
                        transitions.push_back(std::vector<std::map<uint_fast64_t, ValueType>>());
                        for (auto i = 0; i < numberOfActions; i++) {
                            transitions[nextNewStateIndex].push_back(std::map<uint_fast64_t, ValueType>());
                            choiceCount++;
                        }
                        nextNewStateIndex++;
                        processingQ.push(succEpochs);
                    }
                    // add transition to that state
                    transitions[stateEpochsToNewState[currentStateEpochsPair]][actionIndex][unfSuccState] = entry.getValue();
                    entryCount++;
                }
            }
        }
    }

    // Observations
    observations.push_back(originalPomdp->getNrObservations());      // =)
    observations.push_back(originalPomdp->getNrObservations() + 1);  // =(
    for (uint_fast64_t i = 2; i < nextNewStateIndex; i++) {
        observations.push_back(originalPomdp->getObservation(newStateToStateEpoch[i].first));
    }

    // State labeling: single label for =)
    auto stateLabeling = storm::models::sparse::StateLabeling(stateEpochsToNewState.size() + 2);
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
    auto components = storm::storage::sparse::ModelComponents(std::move(unfoldedTransitionMatrix), std::move(stateLabeling));
    components.observabilityClasses = observations;

    // Optional copy of choice labels
    if (originalPomdp->hasChoiceLabeling()){
        auto newChoiceLabeling = storm::models::sparse::ChoiceLabeling(choiceCount);
        auto oldChoiceLabeling = originalPomdp->getChoiceLabeling();
        auto newRowGroupIndices = components.transitionMatrix.getRowGroupIndices();
        auto oldRowGroupIndices = originalPomdp->getTransitionMatrix().getRowGroupIndices();

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
    if (originalPomdp->isCanonic()){
        unfoldedPomdp.setIsCanonic();
    }

    // Drop Bounds from Until Formula
    auto vis = storm::logic::BoundToUnboundVisitor();
    auto newFormula = vis.dropBounds(formula);

    /*std::cout << "\nUNFOLDED POMDP:\n";
    pomdp->writeDotToStream(std::cout);
    std::cout << "\nNEW FORMULA:\n";
    newFormula->writeToStream(std::cout);*/

    // Put result together
    return UnfoldingResult(std::make_shared<storm::models::sparse::Pomdp<ValueType>>(std::move(unfoldedPomdp)), newFormula, std::move(stateEpochsToNewState), std::move(newStateToStateEpoch));
}

template<>
double BoundUnfolder<double>::getUpperBound(const storm::logic::BoundedUntilFormula& formula, uint64_t i) {
    return formula.getUpperBound(i).evaluateAsDouble();
}

template<>
storm::RationalNumber BoundUnfolder<storm::RationalNumber>::getUpperBound(const storm::logic::BoundedUntilFormula& formula, uint64_t i) {
    return formula.getUpperBound(i).evaluateAsRational();
}

template<>
double BoundUnfolder<double>::getLowerBound(const storm::logic::BoundedUntilFormula& formula, uint64_t i) {
    return formula.getLowerBound(i).evaluateAsDouble();
}

template<>
storm::RationalNumber BoundUnfolder<storm::RationalNumber>::getLowerBound(const storm::logic::BoundedUntilFormula& formula, uint64_t i) {
    return formula.getLowerBound(i).evaluateAsRational();
}

template<typename ValueType>
std::pair<std::map<std::string, ValueType>, std::map<std::string, ValueType>> BoundUnfolder<ValueType>::getBounds(const logic::Formula &formula) {
    STORM_LOG_ASSERT(formula.isOperatorFormula() && formula.asOperatorFormula().getSubformula().isBoundedUntilFormula(), "Formula is not the right kind");
    auto buFormula = formula.asOperatorFormula().getSubformula().asBoundedUntilFormula();
    std::map<std::string, ValueType> upperBounds;
    std::map<std::string, ValueType> lowerBounds;

    for (uint64_t i = 0; i < buFormula.getDimension(); i++) {
        STORM_LOG_ASSERT(buFormula.getTimeBoundReference(i).hasRewardModelName(), "The reward model has no name");
        std::string rewName = buFormula.getTimeBoundReference(i).getRewardModelName();
        if (buFormula.hasUpperBound(i)) {
            STORM_LOG_ASSERT(buFormula.hasIntegerUpperBound(i), "Bound not integer"); //TODO maybe make this just a warn?
            ValueType ub;
            if (buFormula.isUpperBoundStrict(i)) {
                // Convert strict to non-strict
                ub = getUpperBound(buFormula, i) - storm::utility::one<ValueType>();
            } else {
                // already is non-strict
                ub = getUpperBound(buFormula, i);
            }
            if (upperBounds.find(rewName) == upperBounds.end() || upperBounds[rewName] > ub) {
                // no upper bound for this reward structure exists yet or the one we have is tighter
                upperBounds[rewName] = ub;
            }
        }
        if (buFormula.hasLowerBound(i)) {
            STORM_LOG_ASSERT(buFormula.hasIntegerLowerBound(i), "Bound not integer"); //TODO maybe make this just a warn?
            ValueType lb;
            if (buFormula.isLowerBoundStrict(i)) {
                // Convert strict to non-strict
                lb = getLowerBound(buFormula, i) + storm::utility::one<ValueType>();
            } else {
                // already is non-strict
                lb = getLowerBound(buFormula, i);
            }
            if (lowerBounds.find(rewName) == lowerBounds.end() || lowerBounds[rewName] < lb) {
                // no lower bound for this reward structure exists yet or the one we have is tighter
                lowerBounds[rewName] = lb;
            }
        }
    }
    return std::make_pair(upperBounds, lowerBounds);//pair<std::map<storm::logic::TimeBoundReference, ValueType>, std::map<storm::logic::TimeBoundReference, ValueType>>();
}

template class BoundUnfolder<double>;
template class BoundUnfolder<storm::RationalNumber>;
}  // namespace transformer
}  // namespace storm