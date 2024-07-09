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

namespace storm::transformer {
template<typename ValueType>
typename BoundUnfolder<ValueType>::UnfoldingResult BoundUnfolder<ValueType>::unfold(
    std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> originalPomdp, const storm::logic::Formula& formula) {
    STORM_LOG_ASSERT(originalPomdp->getInitialStates().getNumberOfSetBits() == 1, "Original POMDP has more than one initial state");

    // Check formula
    STORM_LOG_THROW(formula.isProbabilityOperatorFormula() && formula.asOperatorFormula().getSubformula().isBoundedUntilFormula() &&
                        formula.asOperatorFormula().getSubformula().asBoundedUntilFormula().getLeftSubformula().isTrueFormula(),
                    storm::exceptions::NotSupportedException, "Unexpected formula type of formula " << formula);

    // TODO check that reward models are state action rewards
    // STORM_LOG_THROW(rewModel.hasStateActionRewards(), storm::exceptions::NotSupportedException, "Only state action rewards are currently supported.");

    std::vector<std::unordered_map<std::string, ValueType>> idToEpochMap;
    // Grab bounds
    std::unordered_map<std::string, ValueType> upperBounds, lowerBounds;
    std::tie(upperBounds, lowerBounds) = getBounds(formula);

    // Grab matrix (mostly for coding convenience to just have it in a variable here)
    auto& ogMatrix = originalPomdp->getTransitionMatrix();

    // Grab goal states
    storm::pomdp::analysis::FormulaInformation formulaInfo = storm::pomdp::analysis::getFormulaInformation(*originalPomdp, formula);
    storm::storage::BitVector targetStates = formulaInfo.getTargetStates().states;

    // Transformation information + variables (remove non-necessary ones later)
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> stateEpochToNewState;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> newStateToStateEpoch;

    std::queue<std::pair<uint64_t, uint64_t>> processingQ;  // queue does BFS, if DFS is desired, change to stack

    // Information for unfolded model
    // Special sink states: ID 0 is target, ID 1 is sink
    std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> transitions{{{{0ul, storm::utility::one<ValueType>()}}},
                                                                                  {{{1ul, storm::utility::one<ValueType>()}}}};
    // first index (vec): origin state, second index(vec): action, third index(map): destination state, value(map):probability
    std::vector<uint32_t> observations;

    uint64_t nextNewStateIndex = 2;
    uint64_t entryCount = 2;
    uint64_t choiceCount = 2;

    // Create init state of unfolded model
    uint64_t initState = originalPomdp->getInitialStates().getNextSetIndex(0);
    std::unordered_map<std::string, ValueType> initEpoch;
    for (const auto& rewBound : upperBounds) {
        initEpoch[rewBound.first + "_ub"] = rewBound.second;
    }
    for (const auto& rewBound : lowerBounds) {
        initEpoch[rewBound.first + "_lb"] = rewBound.second;
    }
    idToEpochMap.push_back(initEpoch);

    std::pair<uint64_t, uint64_t> initStateEpoch = std::make_pair(initState, 0ul);
    processingQ.push(initStateEpoch);
    uint64_t numberOfActions = ogMatrix.getRowGroupSize(initState);
    transitions.push_back(std::vector<std::unordered_map<uint64_t, ValueType>>());
    for (uint64_t i = 0ul; i < numberOfActions; ++i) {
        transitions[nextNewStateIndex].push_back(std::unordered_map<uint64_t, ValueType>());
        choiceCount++;
    }
    stateEpochToNewState[initState][0ul] = nextNewStateIndex;
    newStateToStateEpoch[nextNewStateIndex] = initStateEpoch;
    ++nextNewStateIndex;

    while (!processingQ.empty()) {
        uint64_t currentOriginalState, currentEpoch;
        std::tie(currentOriginalState, currentEpoch) = processingQ.front();
        processingQ.pop();
        uint64_t rowGroupStart = ogMatrix.getRowGroupIndices().at(currentOriginalState);
        uint64_t rowGroupSize = ogMatrix.getRowGroupSize(currentOriginalState);
        for (uint64_t actionIndex = 0ul; actionIndex < rowGroupSize; actionIndex++) {
            uint64_t row = rowGroupStart + actionIndex;
            std::unordered_map<std::string, ValueType> epochValues;

            // Collect epoch values of upper bounds
            bool upperBoundViolated = false;
            for (auto const& ub : upperBounds) {
                ValueType reward = originalPomdp->getRewardModel(ub.first).getStateActionReward(row);
                if (reward > idToEpochMap.at(currentEpoch)
                                 .at(ub.first + "_ub")) {  // 0 for upper bounds means we can still keep going as long as we dont collect any more of the reward
                    // Entire action goes to sink with prob 1
                    transitions[stateEpochToNewState[currentOriginalState][currentEpoch]][actionIndex][1] = storm::utility::one<ValueType>();
                    ++entryCount;
                    upperBoundViolated = true;  // for going to next action
                    break;
                } else {
                    epochValues[ub.first + "_ub"] = idToEpochMap.at(currentEpoch).at(ub.first + "_ub") - reward;
                }
            }
            if (upperBoundViolated) {
                continue;
            }

            // Collect epoch values of lower bounds
            uint64_t fulfilledLowerBounds = 0;
            for (auto const& lb : lowerBounds) {
                ValueType reward = originalPomdp->getRewardModel(lb.first).getStateActionReward(row);
                if (reward >= idToEpochMap.at(currentEpoch).at(lb.first + "_lb")) {  // 0 for lower bounds means we have fulfilled the bound
                    epochValues[lb.first + "_lb"] = storm::utility::zero<ValueType>();
                    fulfilledLowerBounds++;
                } else {
                    epochValues[lb.first + "_lb"] = idToEpochMap.at(currentEpoch).at(lb.first + "_lb") - reward;
                }
            }
            bool lowerBoundsFulfilled = fulfilledLowerBounds == lowerBounds.size();

            // Get epoch ID, add new if not found
            uint64_t succEpochId;
            auto it = find(idToEpochMap.begin(), idToEpochMap.end(), epochValues);
            // Epoch found
            if (it != idToEpochMap.end()) {
                succEpochId = it - idToEpochMap.begin();
            } else {
                succEpochId = idToEpochMap.size();
                idToEpochMap.push_back(epochValues);
            }

            // Per transition
            for (auto const& entry : ogMatrix.getRow(row)) {
                // get successor in original POMDP
                uint64_t originalSuccState = entry.getColumn();
                if (targetStates[originalSuccState] && lowerBoundsFulfilled) {
                    // transition to target with prob. of the successor (but check if there already is a transition going to target and if so, just add to it)
                    if (transitions.at(stateEpochToNewState.at(currentOriginalState).at(currentEpoch)).at(actionIndex).count(0) == 0) {
                        transitions.at(stateEpochToNewState.at(currentOriginalState).at(currentEpoch)).at(actionIndex)[0] = entry.getValue();
                        ++entryCount;
                    } else {
                        transitions.at(stateEpochToNewState.at(currentOriginalState).at(currentEpoch)).at(actionIndex)[0] += entry.getValue();
                    }
                } else {
                    // see if successor state in unfolding exists already
                    // if not, create it + add it to the queue
                    uint64_t unfoldingSuccState;

                    bool succStateExists =
                        stateEpochToNewState.count(originalSuccState) != 0 && stateEpochToNewState.at(originalSuccState).count(succEpochId) != 0;
                    if (succStateExists) {
                        unfoldingSuccState = stateEpochToNewState.at(originalSuccState).at(succEpochId);
                    } else {
                        unfoldingSuccState = nextNewStateIndex;
                        stateEpochToNewState[originalSuccState][succEpochId] = nextNewStateIndex;
                        newStateToStateEpoch[nextNewStateIndex] = {originalSuccState, succEpochId};
                        numberOfActions = ogMatrix.getRowGroupSize(originalSuccState);
                        transitions.push_back(std::vector<std::unordered_map<uint64_t, ValueType>>());
                        for (uint64_t i = 0ul; i < numberOfActions; i++) {
                            transitions[nextNewStateIndex].push_back(std::unordered_map<uint64_t, ValueType>());
                            ++choiceCount;
                        }
                        ++nextNewStateIndex;
                        processingQ.emplace(originalSuccState, succEpochId);
                    }
                    // add transition to that state
                    transitions.at(stateEpochToNewState.at(currentOriginalState).at(currentEpoch)).at(actionIndex)[unfoldingSuccState] = entry.getValue();
                    ++entryCount;
                }
            }
        }
    }

    // Observations
    observations.push_back(originalPomdp->getNrObservations());      // target
    observations.push_back(originalPomdp->getNrObservations() + 1);  // sink
    for (uint64_t i = 2ul; i < nextNewStateIndex; i++) {
        observations.push_back(originalPomdp->getObservation(newStateToStateEpoch[i].first));
    }

    // State labeling: single label for target
    auto stateLabeling = storm::models::sparse::StateLabeling(stateEpochToNewState.size() + 2);
    auto labeling = storm::storage::BitVector(nextNewStateIndex, false);
    labeling.set(0);
    stateLabeling.addLabel("goal", labeling);
    labeling = storm::storage::BitVector(nextNewStateIndex, false);
    labeling.set(2);
    stateLabeling.addLabel("init", labeling);

    // Build Matrix
    storm::storage::SparseMatrixBuilder<ValueType> builder(choiceCount, nextNewStateIndex, entryCount, true, true, nextNewStateIndex);
    uint64_t nextMatrixRow = 0;
    for (uint64_t state = 0ul; state < transitions.size(); state++) {
        builder.newRowGroup(nextMatrixRow);
        for (uint64_t action = 0ul; action < transitions[state].size(); action++) {
            for (auto const& entry : transitions[state][action]) {
                builder.addNextValue(nextMatrixRow, entry.first, entry.second);
            }
            nextMatrixRow++;
        }
    }

    // Build components
    auto components = storm::storage::sparse::ModelComponents(builder.build(), std::move(stateLabeling));
    components.observabilityClasses = observations;

    // Optional copy of choice labels
    if (originalPomdp->hasChoiceLabeling()) {
        auto newChoiceLabeling = storm::models::sparse::ChoiceLabeling(choiceCount);
        auto oldChoiceLabeling = originalPomdp->getChoiceLabeling();
        std::vector<uint64_t> newRowGroupIndices = components.transitionMatrix.getRowGroupIndices();
        std::vector<uint64_t> oldRowGroupIndices = originalPomdp->getTransitionMatrix().getRowGroupIndices();
        for (uint64_t newState = 2ul; newState < transitions.size(); newState++) {
            uint64_t oldState = newStateToStateEpoch[newState].first;
            uint64_t oldChoiceIndex = oldRowGroupIndices[oldState];
            uint64_t newChoiceIndex = newRowGroupIndices[newState];
            for (uint64_t action = 0ul; action < transitions[newState].size(); action++) {
                for (auto const& label : oldChoiceLabeling.getLabelsOfChoice(oldChoiceIndex + action)) {
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
    if (originalPomdp->isCanonic()) {
        unfoldedPomdp.setIsCanonic();
    }

    // Drop Bounds from Until Formula
    storm::logic::BoundToUnboundVisitor vis;
    std::shared_ptr<storm::logic::Formula> newFormula = vis.dropBounds(formula);

    // Put result together
    return UnfoldingResult(std::make_shared<storm::models::sparse::Pomdp<ValueType>>(std::move(unfoldedPomdp)), newFormula, idToEpochMap, stateEpochToNewState,
                           newStateToStateEpoch);
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
std::pair<std::unordered_map<std::string, ValueType>, std::unordered_map<std::string, ValueType>> BoundUnfolder<ValueType>::getBounds(
    const logic::Formula& formula) {
    STORM_LOG_ASSERT(formula.isOperatorFormula() && formula.asOperatorFormula().getSubformula().isBoundedUntilFormula(),
                     "Formula is not the right kind (Operator Formula with one bounded Until subformula)");
    auto buFormula = formula.asOperatorFormula().getSubformula().asBoundedUntilFormula();
    std::unordered_map<std::string, ValueType> upperBounds;
    std::unordered_map<std::string, ValueType> lowerBounds;

    for (uint64_t i = 0; i < buFormula.getDimension(); i++) {
        STORM_LOG_ASSERT(buFormula.getTimeBoundReference(i).hasRewardModelName(), "The reward model for bound reference " << i << " has no name.");
        std::string rewName = buFormula.getTimeBoundReference(i).getRewardModelName();
        if (buFormula.hasUpperBound(i)) {
            STORM_LOG_ASSERT(buFormula.hasIntegerUpperBound(i), "Bound " << i << " is not integer");
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
            STORM_LOG_ASSERT(buFormula.hasIntegerLowerBound(i), "Bound " << i << " is not integer");
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
    return std::make_pair(upperBounds, lowerBounds);
}

template class BoundUnfolder<double>;
template class BoundUnfolder<storm::RationalNumber>;
}  // namespace storm::transformer
