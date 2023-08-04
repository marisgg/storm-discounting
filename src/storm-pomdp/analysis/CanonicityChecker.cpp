#include "CanonicityChecker.h"
namespace storm {
    namespace pomdp {
        namespace analysis {

            template<typename ValueType>
            CanonicityChecker<ValueType>::CanonicityChecker() {}

            template<typename ValueType>
            bool CanonicityChecker<ValueType>::check(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp) {
                if (pomdp->hasChoiceLabeling()) {
                    return checkWithLabels(pomdp);
                } else {
                    return checkWithoutLabels(pomdp);
                }
            }

            template<typename ValueType>
            bool CanonicityChecker<ValueType>::checkWithLabels(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp) {
                auto observationsToActions = std::map<uint_fast64_t, std::vector<std::string>>();
                auto rowGroupIndices = pomdp->getTransitionMatrix().getRowGroupIndices();
                auto choiceLabeling = pomdp->getChoiceLabeling();
                auto numberOfStates = pomdp->getNumberOfStates();

                for (uint_fast64_t state = 0; state < numberOfStates; state++){
                    // Grab observation and collect labels
                    uint_fast64_t observation = pomdp->getObservation(state);
                    auto labels = std::vector<std::string>();
                    for (auto row = rowGroupIndices[state]; row < pomdp->getTransitionMatrix().getRowCount() && row < rowGroupIndices[state + 1]; row++) {
                        auto choiceLabelSet = choiceLabeling.getLabelsOfChoice(row);
                        if (choiceLabelSet.empty()) {
                            labels.push_back("");
                        } else if (choiceLabelSet.size() == 1) {
                            labels.push_back(*(choiceLabeling.getLabelsOfChoice(row).begin()));
                        } else {
                            STORM_LOG_ERROR("There are multiple action labels for the same action");
                        }
                    }

                    if (observationsToActions.find(observation) == observationsToActions.end()) {
                        // Nothing saved for this observation yet
                        observationsToActions[observation] = std::move(labels);
                    } else {
                        // Something saved already, compare length (number of actions) and labels (action names)
                        if (observationsToActions[observation].size() != labels.size()) {
                            return false;
                        }
                        for (auto i = 0; i < labels.size(); i++) {
                            if (observationsToActions[observation][i] != labels[i]) {
                                return false;
                            }
                        }
                    }
                }
                return true;
            }

            template<typename ValueType>
            bool CanonicityChecker<ValueType>::checkWithoutLabels(std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> pomdp) {
                auto observationsToActionNumber = std::map<uint_fast64_t, uint_fast64_t>();
                auto rowGroupIndices = pomdp->getTransitionMatrix().getRowGroupIndices();
                auto numberOfStates = pomdp->getNumberOfStates();

                for (uint_fast64_t state = 0; state < numberOfStates; state++){
                    // Grab observation and calculate number of actions
                    uint_fast64_t observation = pomdp->getObservation(state);
                    uint_fast64_t numberOfActions;
                    if (state < numberOfStates - 1) {
                        numberOfActions = rowGroupIndices[state + 1] - rowGroupIndices[state];
                    } else {
                        numberOfActions = pomdp->getTransitionMatrix().getRowCount() - rowGroupIndices[state];
                    }

                    if (observationsToActionNumber.find(observation) == observationsToActionNumber.end()){
                        // Nothing saved for this observation yet
                        observationsToActionNumber[observation] = numberOfActions;
                    } else if (observationsToActionNumber[observation] != numberOfActions) {
                        // Something saved already, and it is not the same amount of actions
                        return false;
                    }
                }
                return true;
            }

            template
            class CanonicityChecker<double>;

            template
            class CanonicityChecker<storm::RationalNumber>;
        }
    }
}