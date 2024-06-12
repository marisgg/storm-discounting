#pragma once

#include "BeliefBasedModelCheckerOptions.h"
#include "storm-pomdp/beliefs/verification/PropertyInformation.h"
#include "storm-pomdp/modelchecker/BeliefExplorationPomdpModelCheckerOptions.h"
#include "storm-pomdp/storage/BeliefExplorationBounds.h"

namespace storm {
class Environment;

namespace pomdp::beliefs {

template<typename PomdpModelType, typename BeliefValueType = typename PomdpModelType::ValueType,
         typename BeliefMdpValueType = typename PomdpModelType::ValueType>
class BeliefBasedModelChecker {
   public:
    explicit BeliefBasedModelChecker(PomdpModelType const& pomdp);

    std::pair<BeliefMdpValueType, bool> checkUnfold(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                    storm::pomdp::storage::PreprocessingPomdpValueBounds<BeliefMdpValueType> const& valueBounds);

    std::pair<BeliefMdpValueType, bool> checkDiscretize(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                        storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                        uint64_t resolution, bool useDynamic,
                                                        storm::pomdp::storage::PreprocessingPomdpValueBounds<BeliefMdpValueType> const& valueBounds);

    std::pair<BeliefMdpValueType, bool> checkRewardAwareUnfold(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                               storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                               storm::pomdp::storage::PreprocessingPomdpValueBounds<BeliefMdpValueType> const& valueBounds,
                                                               std::vector<std::string> const& relevantRewardModelNames = {});

    std::pair<BeliefMdpValueType, bool> checkRewardAwareDiscretize(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                                   storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                                   uint64_t resolution, bool useDynamic,
                                                                   storm::pomdp::storage::PreprocessingPomdpValueBounds<BeliefMdpValueType> const& valueBounds,
                                                                   std::vector<std::string> const& relevantRewardModelNames = {});

   private:
    PomdpModelType const& inputPomdp;
};
}  // namespace pomdp::beliefs
}  // namespace storm