#pragma once

#include <functional>
#include <memory>

#include "storm-pomdp/beliefs/exploration/ExplorationInformation.h"
#include "storm-pomdp/beliefs/verification/PropertyInformation.h"
#include "storm/logic/Formulas.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"

namespace storm::pomdp::beliefs {

std::shared_ptr<storm::logic::Formula const> createFormulaForBeliefMdp(PropertyInformation const& propertyInformation);

// TODO: overloads for extra transition data (e.g. reward vectors)

template<typename BeliefMdpValueType, typename BeliefType, typename... ExtraTransitionData>
std::shared_ptr<storm::models::sparse::Mdp<BeliefMdpValueType>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<BeliefMdpValueType, BeliefType, ExtraTransitionData...> const& explorationInformation,
    PropertyInformation const& propertyInformation, std::function<BeliefMdpValueType(BeliefType const&)> computeCutOffValue);

/**
 * Variant for explicit cut-offs in frontier beliefs
 * TODO: document
 * @tparam BeliefMdpValueType
 * @tparam BeliefType
 * @param explorationInformation
 * @param propertyInformation
 * @param computeCutOffValueMap
 * @return
 */
template<typename BeliefMdpValueType, typename BeliefType, typename... ExtraTransitionData>
std::shared_ptr<storm::models::sparse::Mdp<BeliefMdpValueType>> buildBeliefMdp(
    ExplorationInformation<BeliefMdpValueType, BeliefType, ExtraTransitionData...> const& explorationInformation,
    PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, BeliefMdpValueType>(BeliefType const&)> computeCutOffValueMap);

}  // namespace storm::pomdp::beliefs