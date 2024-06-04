#include "storm-pomdp/beliefs/exploration/BeliefMdpBuilder.h"

#include "storm-pomdp/beliefs/storage/Belief.h"

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/sparse/ModelComponents.h"

#include "storm/exceptions/UnexpectedException.h"

namespace storm::pomdp::beliefs {

std::shared_ptr<storm::logic::Formula const> createFormulaForBeliefMdp(PropertyInformation const& propertyInformation) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability ||
                         propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward,
                     "Unexpected kind of property.");
    switch (propertyInformation.kind) {
        case PropertyInformation::Kind::ReachabilityProbability: {
            auto target = std::make_shared<storm::logic::AtomicLabelFormula const>("target");
            auto eventuallyTarget = std::make_shared<storm::logic::EventuallyFormula const>(target, storm::logic::FormulaContext::Probability);
            return std::make_shared<storm::logic::ProbabilityOperatorFormula const>(eventuallyTarget,
                                                                                    storm::logic::OperatorInformation(propertyInformation.dir));
        }
        case PropertyInformation::Kind::ExpectedTotalReachabilityReward: {
            auto bottom = std::make_shared<storm::logic::AtomicLabelFormula const>("bottom");
            auto eventuallyBottom = std::make_shared<storm::logic::EventuallyFormula const>(bottom, storm::logic::FormulaContext::Reward,
                                                                                            storm::logic::RewardAccumulation(true, false, false));
            return std::make_shared<storm::logic::RewardOperatorFormula const>(eventuallyBottom, propertyInformation.rewardModelName.value(),
                                                                               storm::logic::OperatorInformation(propertyInformation.dir));
        }
    }
    STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "Unhandled case.");
}

template<typename BeliefMdpValueType, typename BeliefType>
std::shared_ptr<storm::models::sparse::Mdp<BeliefMdpValueType>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<BeliefMdpValueType, BeliefType> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<BeliefMdpValueType(BeliefType const&)> computeCutOffValue) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability ||
                         propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward,
                     "Unexpected kind of property.");
    bool const reachabilityProbability = propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability;
    uint64_t const numExtraStates = reachabilityProbability ? 2ull : 1ull;
    uint64_t const numStates = explorationInformation.matrix.groups() + numExtraStates;
    uint64_t const numChoices = explorationInformation.matrix.rows() + numExtraStates;
    uint64_t const targetState = numStates - numExtraStates;
    uint64_t const bottomState = numStates - 1;
    std::vector<BeliefMdpValueType> actionRewards;
    if (!reachabilityProbability) {
        actionRewards.reserve(numChoices);
        actionRewards.insert(actionRewards.end(), explorationInformation.actionRewards.begin(), explorationInformation.actionRewards.end());
        actionRewards.push_back(storm::utility::zero<BeliefMdpValueType>());
        STORM_LOG_ASSERT(numChoices == actionRewards.size(),
                         "Unexpected size of action rewards: Expected " << numChoices << " got " << actionRewards.size() << ".");
    }
    storm::storage::SparseMatrixBuilder<BeliefMdpValueType> transitionBuilder(numChoices, numStates, 0, true, true, numStates);
    for (uint64_t state = 0; state < numStates - numExtraStates; ++state) {
        uint64_t choice = explorationInformation.matrix.rowGroupIndices[state];
        transitionBuilder.newRowGroup(choice);
        for (uint64_t const groupEnd = explorationInformation.matrix.rowGroupIndices[state + 1]; choice < groupEnd; ++choice) {
            auto probabilityToBottom = storm::utility::zero<BeliefMdpValueType>();
            auto probabilityToTarget = storm::utility::zero<BeliefMdpValueType>();
            for (uint64_t entryIndex = explorationInformation.matrix.rowIndications[choice];
                 entryIndex < explorationInformation.matrix.rowIndications[choice + 1]; ++entryIndex) {
                auto const& entry = explorationInformation.matrix.transitions[entryIndex];
                if (auto explIt = explorationInformation.exploredBeliefs.find(entry.targetBelief); explIt != explorationInformation.exploredBeliefs.end()) {
                    // Transition to explored belief
                    transitionBuilder.addNextValue(choice, explIt->second, entry.probability);
                } else {
                    // Transition to unexplored belief (either terminal or cut-off)
                    BeliefMdpValueType successorValue;
                    if (auto terminalIt = explorationInformation.terminalBeliefValues.find(entry.targetBelief);
                        terminalIt != explorationInformation.terminalBeliefValues.end()) {
                        successorValue = entry.probability * terminalIt->second;  // terminal value determined during exploration
                    } else {
                        // Transition to cut-off belief
                        BeliefType const& successorBelief = explorationInformation.discoveredBeliefs.getBeliefFromId(entry.targetBelief);
                        successorValue = entry.probability * computeCutOffValue(successorBelief);  // Cut-off value
                    }
                    if (reachabilityProbability) {
                        probabilityToTarget += successorValue;
                        probabilityToBottom += entry.probability - successorValue;
                    } else {
                        probabilityToBottom += entry.probability;
                        actionRewards[choice] += successorValue;
                    }
                }
            }
            if (reachabilityProbability && !storm::utility::isZero(probabilityToTarget)) {
                transitionBuilder.addNextValue(choice, targetState, probabilityToTarget);
            }
            if (!storm::utility::isZero(probabilityToBottom)) {
                transitionBuilder.addNextValue(choice, bottomState, probabilityToBottom);
            }
        }
    }
    if (reachabilityProbability) {
        transitionBuilder.newRowGroup(numChoices - 2);
        transitionBuilder.addNextValue(numChoices - 2, targetState, storm::utility::one<BeliefMdpValueType>());
    }
    transitionBuilder.newRowGroup(numChoices - 1);
    transitionBuilder.addNextValue(numChoices - 1, bottomState, storm::utility::one<BeliefMdpValueType>());

    storm::models::sparse::StateLabeling stateLabeling(numStates);
    stateLabeling.addLabel("bottom");
    stateLabeling.addLabelToState("bottom", bottomState);
    stateLabeling.addLabel("init");
    stateLabeling.addLabelToState("init", explorationInformation.exploredBeliefs.at(explorationInformation.initialBeliefId));

    if (reachabilityProbability) {
        stateLabeling.addLabel("target");
        stateLabeling.addLabelToState("target", targetState);
    }
    storm::storage::sparse::ModelComponents<BeliefMdpValueType> components(transitionBuilder.build(), std::move(stateLabeling));

    if (!reachabilityProbability) {
        storm::models::sparse::StandardRewardModel<BeliefMdpValueType> rewardModel(std::nullopt, std::move(actionRewards));
        components.rewardModels.emplace(propertyInformation.rewardModelName.value(), std::move(rewardModel));
    }

    return std::make_shared<storm::models::sparse::Mdp<BeliefMdpValueType>>(std::move(components));
}

template<typename BeliefMdpValueType, typename BeliefType>
std::shared_ptr<storm::models::sparse::Mdp<BeliefMdpValueType>> buildBeliefMdp(
    ExplorationInformation<BeliefMdpValueType, BeliefType> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, BeliefMdpValueType>(BeliefType const&)> computeCutOffValueMap) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability ||
                         propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward,
                     "Unexpected kind of property.");
    bool const reachabilityProbability = propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability;

    // First gather all cut-off information
    uint64_t nrCutOffChoices = 0ull;
    std::unordered_map<BeliefId, std::unordered_map<std::string, BeliefMdpValueType>> cutOffInformationMap;
    for (auto const& frontierBeliefId : explorationInformation.frontierBeliefs) {
        auto const& frontierBelief = explorationInformation.discoveredBeliefs.getBeliefFromId(frontierBeliefId);
        cutOffInformationMap[frontierBeliefId] = computeCutOffValueMap(frontierBelief);
        nrCutOffChoices += cutOffInformationMap[frontierBeliefId].size();
    }

    uint64_t const numBottomTargetStates = reachabilityProbability ? 2ull : 1ull;
    uint64_t const numExtraStates = numBottomTargetStates + explorationInformation.frontierBeliefs.size();
    uint64_t const numStates = explorationInformation.matrix.groups() + numExtraStates;
    uint64_t const numChoices = explorationInformation.matrix.rows() + numBottomTargetStates + nrCutOffChoices;
    uint64_t const targetState = numStates - (reachabilityProbability ? 2ull : 1ull);
    uint64_t const bottomState = numStates - 1;

    std::vector<BeliefMdpValueType> actionRewards;
    if (!reachabilityProbability) {
        actionRewards.reserve(numChoices);
        actionRewards.insert(actionRewards.end(), explorationInformation.actionRewards.begin(), explorationInformation.actionRewards.end());
        // Insert 0 for all cut-off choices and bottom state
        actionRewards.insert(actionRewards.end(), nrCutOffChoices + 1ull, storm::utility::zero<BeliefMdpValueType>());
        STORM_LOG_ASSERT(numChoices == actionRewards.size(),
                         "Unexpected size of action rewards: Expected " << numChoices << " got " << actionRewards.size() << ".");
    }

    std::unordered_map<BeliefId, uint64_t> frontierBeliefToStateMap;
    std::unordered_map<uint64_t, BeliefId> stateToFrontierBeliefMap;
    uint64_t nextStateId = numStates - numExtraStates;

    storm::storage::SparseMatrixBuilder<BeliefMdpValueType> transitionBuilder(numChoices, numStates, 0, true, true, numStates);
    // Treat explored beliefs
    for (uint64_t state = 0; state < numStates - numExtraStates; ++state) {
        uint64_t choice = explorationInformation.matrix.rowGroupIndices[state];
        transitionBuilder.newRowGroup(choice);
        for (uint64_t const groupEnd = explorationInformation.matrix.rowGroupIndices[state + 1]; choice < groupEnd; ++choice) {
            auto probabilityToBottom = storm::utility::zero<BeliefMdpValueType>();
            auto probabilityToTarget = storm::utility::zero<BeliefMdpValueType>();
            for (uint64_t entryIndex = explorationInformation.matrix.rowIndications[choice];
                 entryIndex < explorationInformation.matrix.rowIndications[choice + 1]; ++entryIndex) {
                auto const& entry = explorationInformation.matrix.transitions[entryIndex];
                if (auto explIt = explorationInformation.exploredBeliefs.find(entry.targetBelief); explIt != explorationInformation.exploredBeliefs.end()) {
                    // Transition to explored belief
                    transitionBuilder.addNextValue(choice, explIt->second, entry.probability);
                } else {
                    // Transition to unexplored belief (either terminal or cut-off)
                    BeliefMdpValueType successorValue;
                    if (auto terminalIt = explorationInformation.terminalBeliefValues.find(entry.targetBelief);
                        terminalIt != explorationInformation.terminalBeliefValues.end()) {
                        successorValue = entry.probability * terminalIt->second;  // terminal value determined during exploration
                        if (reachabilityProbability) {
                            probabilityToTarget += successorValue;
                            probabilityToBottom += entry.probability - successorValue;
                        } else {
                            probabilityToBottom += entry.probability;
                            actionRewards[choice] += successorValue;
                        }
                    } else {
                        // Transition to frontier belief
                        auto [insertIterator, inserted] = frontierBeliefToStateMap.insert({entry.targetBelief, nextStateId});
                        if (inserted) {
                            stateToFrontierBeliefMap[nextStateId] = entry.targetBelief;
                            ++nextStateId;
                        }
                        transitionBuilder.addNextValue(choice, insertIterator->second, entry.probability);
                    }
                }
            }
            if (reachabilityProbability && !storm::utility::isZero(probabilityToTarget)) {
                transitionBuilder.addNextValue(choice, targetState, probabilityToTarget);
            }
            if (!storm::utility::isZero(probabilityToBottom)) {
                transitionBuilder.addNextValue(choice, bottomState, probabilityToBottom);
            }
        }
    }
    // Treat frontier beliefs
    uint64_t choice = explorationInformation.matrix.rows();
    for (uint64_t state = numStates - numExtraStates; state < numStates - numBottomTargetStates; ++state) {
        transitionBuilder.newRowGroup(choice);
        std::unordered_map<std::string, BeliefMdpValueType> cutOffInformationForBelief = cutOffInformationMap.at(stateToFrontierBeliefMap.at(state));
        for (auto const& entry : cutOffInformationForBelief) {
            if (reachabilityProbability) {
                transitionBuilder.addNextValue(choice, targetState, entry.second);
                transitionBuilder.addNextValue(choice, bottomState, storm::utility::one<BeliefMdpValueType>() - entry.second);
            } else {
                transitionBuilder.addNextValue(choice, bottomState, storm::utility::one<BeliefMdpValueType>());
                actionRewards[choice] += entry.second;
            }
            // TODO add labeling information
            ++choice;
        }
    }

    // Treat extra states
    if (reachabilityProbability) {
        transitionBuilder.newRowGroup(numChoices - 2);
        transitionBuilder.addNextValue(numChoices - 2, targetState, storm::utility::one<BeliefMdpValueType>());
    }
    transitionBuilder.newRowGroup(numChoices - 1);
    transitionBuilder.addNextValue(numChoices - 1, bottomState, storm::utility::one<BeliefMdpValueType>());

    storm::models::sparse::StateLabeling stateLabeling(numStates);
    stateLabeling.addLabel("bottom");
    stateLabeling.addLabelToState("bottom", bottomState);
    stateLabeling.addLabel("init");
    stateLabeling.addLabelToState("init", explorationInformation.exploredBeliefs.at(explorationInformation.initialBeliefId));

    if (reachabilityProbability) {
        stateLabeling.addLabel("target");
        stateLabeling.addLabelToState("target", targetState);
    }
    storm::storage::sparse::ModelComponents<BeliefMdpValueType> components(transitionBuilder.build(), std::move(stateLabeling));

    if (!reachabilityProbability) {
        storm::models::sparse::StandardRewardModel<BeliefMdpValueType> rewardModel(std::nullopt, std::move(actionRewards));
        components.rewardModels.emplace(propertyInformation.rewardModelName.value(), std::move(rewardModel));
    }

    return std::make_shared<storm::models::sparse::Mdp<BeliefMdpValueType>>(std::move(components));
}

template std::shared_ptr<storm::models::sparse::Mdp<double>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<double, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<double(Belief<double> const&)> computeCutOffValue);

template std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<storm::RationalNumber, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<storm::RationalNumber(Belief<double> const&)> computeCutOffValue);

template std::shared_ptr<storm::models::sparse::Mdp<double>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<double, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<double(Belief<storm::RationalNumber> const&)> computeCutOffValue);

template std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<storm::RationalNumber, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<storm::RationalNumber(Belief<storm::RationalNumber> const&)> computeCutOffValue);

template std::shared_ptr<storm::models::sparse::Mdp<double>> buildBeliefMdp(
    ExplorationInformation<double, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, double>(Belief<double> const&)> computeCutOffValueMap);

template std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>> buildBeliefMdp(
    ExplorationInformation<storm::RationalNumber, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, storm::RationalNumber>(Belief<double> const&)> computeCutOffValueMap);

template std::shared_ptr<storm::models::sparse::Mdp<double>> buildBeliefMdp(
    ExplorationInformation<double, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, double>(Belief<storm::RationalNumber> const&)> computeCutOffValueMap);

template std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>> buildBeliefMdp(
    ExplorationInformation<storm::RationalNumber, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, storm::RationalNumber>(Belief<storm::RationalNumber> const&)> computeCutOffValueMap);

}  // namespace storm::pomdp::beliefs