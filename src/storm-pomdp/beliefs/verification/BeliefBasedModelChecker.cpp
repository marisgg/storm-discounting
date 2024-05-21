#include "storm-pomdp/beliefs/verification/BeliefBasedModelChecker.h"

#include <memory>

#include "storm-pomdp/beliefs/abstraction/FreudenthalTriangulationBeliefAbstraction.h"
#include "storm-pomdp/beliefs/exploration/BeliefExploration.h"
#include "storm-pomdp/beliefs/exploration/BeliefMdpBuilder.h"
#include "storm-pomdp/beliefs/storage/Belief.h"

#include "BeliefBasedModelCheckerOptions.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/api/verification.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/models/sparse/Pomdp.h"
#include "storm/utility/OptionalRef.h"
#include "storm/utility/Stopwatch.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm::pomdp::beliefs {

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>

BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::BeliefBasedModelChecker(PomdpModelType const& pomdp) : inputPomdp(pomdp) {
    STORM_LOG_ERROR_COND(inputPomdp.isCanonic(), "Input Pomdp is not known to be canonic. This might lead to unexpected verification results.");
}

template<typename PomdpModelType, typename BeliefType, typename BeliefMdpValueType>
std::pair<BeliefMdpValueType, bool> checkUnfoldOrDiscretize(storm::Environment const& env, PomdpModelType const& pomdp,
                                                            PropertyInformation const& propertyInformation,
                                                            storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                            storm::pomdp::storage::PreprocessingPomdpValueBounds<BeliefMdpValueType> const& valueBounds,
                                           storm::OptionalRef<FreudenthalTriangulationBeliefAbstraction<BeliefType>> abstraction = {}) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability ||
                         propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward,
                     "Unexpected kind of property.");

    STORM_PRINT_AND_LOG("Constructing the belief MDP...\n");

    // First, explore the beliefs and its successors
    using BeliefExplorationType = BeliefExploration<BeliefMdpValueType, PomdpModelType, BeliefType>;
    storm::utility::Stopwatch swExplore(true);
    BeliefExplorationType exploration(pomdp);
    auto info = exploration.initializeStandardExploration();

    // Determine terminationCallback based on options
    typename BeliefExplorationType::TerminationCallback terminationCallback;
    switch (options.getTerminationCriterion()) {
        case MAX_EXPLORATION_SIZE:
            if (options.implicitCutOffs) {
                terminationCallback = [&info, maxSize = options.maxExplorationSize.value()]() { return info.exploredBeliefs.size() > maxSize; };
            } else {
                terminationCallback = [&info, maxSize = options.maxExplorationSize.value()]() {
                    return info.discoveredBeliefs.getNumberOfBeliefIds() > maxSize;
                };
            }
            break;
        case MAX_EXPLORATION_TIME:
            terminationCallback = [&swExplore, maxDuration = options.maxExplorationTime.value()]() {
                return (unsigned)abs(swExplore.getTimeInSeconds()) > maxDuration;
            };
            break;
        case MAX_EXPLORATION_SIZE_AND_TIME:
            if (options.implicitCutOffs) {
                terminationCallback = [&info, &swExplore, maxSize = options.maxExplorationSize.value(), maxDuration = options.maxExplorationTime.value()]() {
                    return info.exploredBeliefs.size() > maxSize || (unsigned)abs(swExplore.getTimeInSeconds()) > maxDuration;
                };
            } else {
                terminationCallback = [&info, &swExplore, maxSize = options.maxExplorationSize.value(), maxDuration = options.maxExplorationTime.value()]() {
                    return info.discoveredBeliefs.getNumberOfBeliefIds() > maxSize || (unsigned)abs(swExplore.getTimeInSeconds()) > maxDuration;
                };
            }
            break;
        case NONE:
            // Unlimited unfolding (useful for known finite belief MDPs)
            terminationCallback = []() { return false; };
            break;
        default:
            STORM_LOG_ERROR("Unknown termination criterion for belief exploration.");
            break;
    }

    // Determine terminalBeliefCallback based on options
    typename BeliefExplorationType::TerminalBeliefCallback terminalBeliefCallback;
    if (propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward) {
        if (options.maxGapToCut.has_value()) {
            // Terminate if the gap is small enough
            terminalBeliefCallback = [&propertyInformation, &valueBounds,
                                      maxGapToCut = options.maxGapToCut.value()](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
                if (propertyInformation.targetObservations.count(belief.observation()) > 0) {
                    return storm::utility::zero<BeliefMdpValueType>();
                } else {
                    // TODO add scheduler information if requested
                    auto smallestUpper = storm::utility::infinity<BeliefMdpValueType>();
                    for (auto const& valueList : valueBounds.upper) {
                        smallestUpper = std::min(smallestUpper, belief.template getWeightedSum<BeliefMdpValueType>(valueList));
                    }
                    BeliefMdpValueType largestLower = -storm::utility::infinity<BeliefMdpValueType>();
                    for (auto const& valueList : valueBounds.lower) {
                        largestLower = std::max(largestLower, belief.template getWeightedSum<BeliefMdpValueType>(valueList));
                    }
                    if (storm::utility::abs<BeliefMdpValueType>(smallestUpper - largestLower) <= maxGapToCut) {
                        return propertyInformation.dir == solver::OptimizationDirection::Maximize ? largestLower : smallestUpper;
                    }
                    return std::nullopt;
                }
            };
        } else {
            terminalBeliefCallback = [&propertyInformation](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
                if (propertyInformation.targetObservations.count(belief.observation()) > 0) {
                    return storm::utility::zero<BeliefMdpValueType>();
                } else {
                    return std::nullopt;
                }
            };
        }
    } else if (options.maxGapToCut.has_value()) {
        // Terminate if the gap is small enough
        terminalBeliefCallback = [&propertyInformation, &valueBounds,
                                  maxGapToCut = options.maxGapToCut.value()](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
            if (propertyInformation.targetObservations.count(belief.observation()) > 0) {
                return storm::utility::one<BeliefMdpValueType>();
            } else {
                // TODO add scheduler information if requested
                auto smallestUpper = storm::utility::infinity<BeliefMdpValueType>();
                for (auto const& valueList : valueBounds.upper) {
                    smallestUpper = std::min(smallestUpper, belief.template getWeightedSum<BeliefMdpValueType>(valueList));
                }
                BeliefMdpValueType largestLower = -storm::utility::infinity<BeliefMdpValueType>();
                for (auto const& valueList : valueBounds.lower) {
                    largestLower = std::max(largestLower, belief.template getWeightedSum<BeliefMdpValueType>(valueList));
                }
                if (storm::utility::abs<BeliefMdpValueType>(smallestUpper - largestLower) <= maxGapToCut) {
                    return propertyInformation.dir == solver::OptimizationDirection::Maximize ? largestLower : smallestUpper;
                }
                return std::nullopt;
            }
        };
    } else {
        terminalBeliefCallback = [&propertyInformation](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
            if (propertyInformation.targetObservations.count(belief.observation()) > 0) {
                return storm::utility::one<BeliefMdpValueType>();
            } else {
                return std::nullopt;
            };
        };
    }

    if (propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward) {
        exploration.resumeExploration(info, terminalBeliefCallback, terminationCallback, propertyInformation.rewardModelName.value(), abstraction);
    } else {
        exploration.resumeExploration(info, terminalBeliefCallback, terminationCallback, storm::NullRef, abstraction);
    }
    swExplore.stop();
    bool earlyExplorationStop = !info.queue.hasNext();
    if (earlyExplorationStop) {
        STORM_PRINT_AND_LOG("Exploration stopped before all states were explored.");
    }

    // Second, build the Belief MDP from the exploration information
    storm::utility::Stopwatch swBuild(true);
    std::shared_ptr<storm::models::sparse::Mdp<BeliefMdpValueType>> beliefMdp;
    if (options.implicitCutOffs) {
        std::function<BeliefMdpValueType(BeliefType const&)> computeCutOffValue = [&valueBounds, &propertyInformation](BeliefType const& belief) {
            // TODO: extend with different sources for cut-offs
            auto result = storm::utility::infinity<BeliefMdpValueType>();
            if (propertyInformation.dir == storm::OptimizationDirection::Minimize) {
                for (auto const& valueList : valueBounds.upper) {
                    result = std::min(result, belief.template getWeightedSum<BeliefMdpValueType>(valueList));
                }
            } else {
                result = -storm::utility::infinity<BeliefMdpValueType>();
                for (auto const& valueList : valueBounds.lower) {
                    result = std::max(result, belief.template getWeightedSum<BeliefMdpValueType>(valueList));
                }
            }
            return result;
        };
        beliefMdp = buildBeliefMdpWithImplicitCutoffs(info, propertyInformation, computeCutOffValue);
    } else {
        std::function<std::unordered_map<std::string, BeliefMdpValueType>(BeliefType const&)> computeCutOffValueMap =
            [&valueBounds, &propertyInformation](BeliefType const& belief) {
                // TODO: extend with different sources for cut-offs
                uint64_t const nrCutoffPolicies =
                    propertyInformation.dir == storm::OptimizationDirection::Minimize ? valueBounds.upper.size() : valueBounds.lower.size();
                std::unordered_map<std::string, BeliefMdpValueType> result;
                for (uint64_t i = 0; i < nrCutoffPolicies; ++i) {
                    auto val = belief.template getWeightedSum<BeliefMdpValueType>(
                        propertyInformation.dir == storm::OptimizationDirection::Minimize ? valueBounds.upper.at(i) : valueBounds.lower.at(i));
                    result["sched_" + std::to_string(i)] = val;
                }
                return result;
            };
        beliefMdp = buildBeliefMdp(info, propertyInformation, computeCutOffValueMap);
    }
    swBuild.stop();
    beliefMdp->printModelInformationToStream(std::cout);

    // Finally, perform model checking on the belief MDP.
    storm::utility::Stopwatch swCheck(true);
    auto formula = createFormulaForBeliefMdp(propertyInformation);
    storm::modelchecker::CheckTask<storm::logic::Formula, BeliefMdpValueType> task(*formula, true);
    std::unique_ptr<storm::modelchecker::CheckResult> res(storm::api::verifyWithSparseEngine<BeliefMdpValueType>(env, beliefMdp, task));
    swCheck.stop();
    STORM_PRINT_AND_LOG("Time for exploring beliefs: " << swExplore << ".\n");
    STORM_PRINT_AND_LOG("Time for building the belief MDP: " << swBuild << ".\n");
    STORM_PRINT_AND_LOG("Time for analyzing the belief MDP: " << swCheck << ".\n");
    STORM_LOG_ASSERT(res, "Model checking of belief MDP did not return any result.");
    STORM_LOG_ASSERT(res->isExplicitQuantitativeCheckResult(), "Model checking of belief MDP did not return result of expected type.");
    STORM_LOG_ASSERT(beliefMdp->getInitialStates().getNumberOfSetBits() == 1, "Unexpected number of initial states for belief Mdp.");
    auto const initState = beliefMdp->getInitialStates().getNextSetIndex(0);
    return {res->asExplicitQuantitativeCheckResult<BeliefMdpValueType>()[initState], !earlyExplorationStop};
}

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
std::pair<BeliefMdpValueType, bool> BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::checkUnfold(
    storm::Environment const& env, PropertyInformation const& propertyInformation,
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
    storm::pomdp::storage::PreprocessingPomdpValueBounds<BeliefMdpValueType> const& valueBounds) {
    return checkUnfoldOrDiscretize<PomdpModelType, Belief<BeliefValueType>, BeliefMdpValueType>(env, inputPomdp, propertyInformation, options, valueBounds);
}

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
std::pair<BeliefMdpValueType, bool> BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::checkDiscretize(
    storm::Environment const& env, PropertyInformation const& propertyInformation,
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options, uint64_t resolution, bool useDynamic,
    storm::pomdp::storage::PreprocessingPomdpValueBounds<BeliefMdpValueType> const& valueBounds) {
    std::vector<BeliefValueType> observationResolutionVector(inputPomdp.getNrObservations(), storm::utility::convertNumber<BeliefValueType>(resolution));
    auto mode = useDynamic ? FreudenthalTriangulationMode::Dynamic : FreudenthalTriangulationMode::Static;
    FreudenthalTriangulationBeliefAbstraction<Belief<BeliefValueType>> abstraction(observationResolutionVector, mode);
    return checkUnfoldOrDiscretize<PomdpModelType, Belief<BeliefValueType>, BeliefMdpValueType>(env, inputPomdp, propertyInformation, options, valueBounds,
                                                                                                abstraction);
}

// TODO: Check which instantiations are actually necessary / reasonable.
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<double>, double, double>;
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<double>, storm::RationalNumber, double>;
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<storm::RationalNumber>, double, storm::RationalNumber>;
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<storm::RationalNumber>, storm::RationalNumber, storm::RationalNumber>;
}  // namespace storm::pomdp::beliefs
