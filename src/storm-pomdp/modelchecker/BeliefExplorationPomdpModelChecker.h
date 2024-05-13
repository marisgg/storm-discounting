#include "storm-pomdp/beliefs/exploration/BeliefMdpBuilder.h"
#include "storm-pomdp/builder/BeliefMdpExplorer.h"
#include "storm-pomdp/modelchecker/BeliefExplorationPomdpModelCheckerOptions.h"
#include "storm-pomdp/storage/BeliefManager.h"
#include "storm/utility/logging.h"

#include "storm/storage/jani/Property.h"
#include "storm/utility/Stopwatch.h"

namespace storm {
class Environment;

namespace models {
namespace sparse {
template<class ValueType, typename RewardModelType>
class Pomdp;
}
}  // namespace models
namespace logic {
class Formula;
}

namespace pomdp {
namespace modelchecker {

/**
 * Structure for storing values on the POMDP used for cut-offs and clipping.
 * trivialPomdpValueBounds is supposed to store
 * extremePomdpValueBound stores the values
 *
 * @tparam ValueType
 */
template<typename ValueType>
struct POMDPValueBounds {
    // values generated by memoryless schedulers during pre-processing
    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> trivialPomdpValueBounds;
    // values for clipping compensation
    storm::pomdp::storage::ExtremePOMDPValueBound<ValueType> extremePomdpValueBound;
    // values generated by a finite memory schedulers. Each scheduler is represented by a vector of maps representing (memory node x state) -> value
    std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> fmSchedulerValueList;
};
/**
 * Model checker for checking reachability queries on POMDPs using approximations based on exploration of the belief MDP
 * @tparam PomdpModelType model type of the POMDP
 * @tparam BeliefValueType type used for the state probabilities in beliefs
 * @tparam BeliefMDPType number type used for the MDP structure of the belief MDP.
 * BeliefMDPType can differ from BeliefValueType as we might want to have exact values for probabilities in beliefs, but are okay with possible imprecision in
 * the MDP itself.
 */
template<typename PomdpModelType, typename BeliefValueType = typename PomdpModelType::ValueType, typename BeliefMDPType = typename PomdpModelType::ValueType>
class BeliefExplorationPomdpModelChecker {
   public:
    typedef BeliefMDPType ValueType;
    typedef typename PomdpModelType::RewardModelType RewardModelType;
    typedef storm::storage::BeliefManager<PomdpModelType, BeliefValueType> BeliefManagerType;
    typedef storm::builder::BeliefMdpExplorer<PomdpModelType, BeliefValueType> ExplorerType;
    typedef BeliefExplorationPomdpModelCheckerOptions<ValueType> Options;

    /* Struct Definition(s) */
    /**
     * Statuses used for the interactive exploration
     */
    enum class Status {
        Uninitialized,
        Exploring,
        ModelExplorationFinished,
        ResultAvailable,
        Terminated,
        Converged,
    };

    /**
     * Struct used to store the results of the model checker
     */
    struct Result {
        Result(ValueType lower, ValueType upper);
        ValueType lowerBound;
        ValueType upperBound;
        ValueType diff(bool relative = false) const;
        bool updateLowerBound(ValueType const& value);
        bool updateUpperBound(ValueType const& value);
        std::shared_ptr<storm::models::sparse::Model<ValueType>> schedulerAsMarkovChain;
        std::vector<storm::storage::Scheduler<ValueType>> cutoffSchedulers;
    };

    /* Functions */

    /**
     * Constructor
     * @param pomdp pointer to the POMDP to be checked
     * @param options object containing the options for the model checker
     */
    explicit BeliefExplorationPomdpModelChecker(std::shared_ptr<PomdpModelType> pomdp, Options options = Options());

    /**
     * Performs model checking of the given POMDP with regards to a formula using the previously specified options
     * @param formula the formula to check
     * @param preProcEnv environment used for solving the pre-processisng
     * @param additionalUnderApproximationBounds additional bounds that can be used for cut-offs in the under-approximation. Each element of the outer vector
     * represents a scheduler. Each scheduler is represented by a vector of maps representing (memory node x state) -> value
     * @return result of the model checking
     */
    Result check(storm::Environment const& env, storm::logic::Formula const& formula, storm::Environment const& preProcEnv,
                 std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> const& additionalUnderApproximationBounds =
                     std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>>());
    Result check(storm::logic::Formula const& formula, storm::Environment const& preProcEnv,
                 std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> const& additionalUnderApproximationBounds =
                     std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>>());
    Result check(storm::logic::Formula const& formula,
                 std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> const& additionalUnderApproximationBounds =
                     std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>>());
    Result check(storm::Environment const& env, storm::logic::Formula const& formula,
                 std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> const& additionalUnderApproximationBounds =
                     std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>>());

    /**
     * Prints statistics of the process to a given output stream
     * @param stream the output stream
     */
    void printStatisticsToStream(std::ostream& stream) const;

    /**
     * Uses model checking on the underlying MDP to generate values used for cut-offs and for clipping compensation if necessary
     * @param formula the formula to check
     * @param preProcEnv environment used for solving the pre-processisng
     */
    void precomputeValueBounds(const logic::Formula& formula, storm::Environment const& preProcEnv);

    /**
     * Allows to generate an under-approximation using a controllable unfolding. This provides a method for outside tools to control the unfolding for an
     * under-approximation themselves. The unfolding runs until a pausing command is issued. If the unfolding is paused, cut-offs and optionally clipping are
     * applied to obtain an abstraction MDP. This MDP is then checked and the result is saved. The unfolding can then be continued from the state before
     * cut-offs were applied.
     * @param targetObservations the target observations of the objective
     * @param min true if the objective is to minimise the value
     * @param rewardModelName name of the reward model to be used if one is specified
     * @param valueBounds values used for cut-offs and clipping
     * @param result the struct to store results
     */
    void unfoldInteractively(storm::Environment const& env, std::set<uint32_t> const& targetObservations, bool min, std::optional<std::string> rewardModelName,
                             storm::pomdp::modelchecker::POMDPValueBounds<ValueType> const& valueBounds, Result& result);
    void unfoldInteractively(std::set<uint32_t> const& targetObservations, bool min, std::optional<std::string> rewardModelName,
                             storm::pomdp::modelchecker::POMDPValueBounds<ValueType> const& valueBounds, Result& result);

    /**
     * Pauses a running interactive unfolding
     */
    void pauseUnfolding();

    /**
     * Continues a previously paused interactive unfolding. Only works if the checking process has already finished and a result is ready.
     */
    void continueUnfolding();

    /**
     * Terminates a running interactive unfolding. Results are computed one last time, then the interactive unfolding is terminated and cannot be continued.
     */
    void terminateUnfolding();

    /**
     * Indicates whether there is a result after an interactive unfolding was paused.
     * @return True, if the model checking process of the current approximation has finished.
     */
    bool isResultReady();

    /**
     * Indicates whether the interactive unfolding is currently in the process of exploring the belief MDP.
     * @return True, if the exploration is currently in progress
     */
    bool isExploring();

    /**
     * Indicates whether the interactive unfolding has coonverged, i.e. it has completely explored a finite belief MDP
     * @return True if the entire belief MDP has been explored
     */
    bool hasConverged();

    /**
     * Get the latest saved result obtained by the interactive unfolding
     * @return
     */
    Result getInteractiveResult();

    /**
     * Get a pointer to the belief explorer used in the interactive unfolding
     * @return pointer to the belief explorer
     */
    std::shared_ptr<ExplorerType> getInteractiveBeliefExplorer();

    /**
     * Get the current status of the interactive unfolding
     * @return the interactive unfolding
     */
    int64_t getStatus();

   private:
    /* Struct Definition(s) */

    /**
     * Control parameters for the interactive unfolding
     */
    enum class UnfoldingControl { Run, Pause, Terminate };

    /**
     * Struct containing statistics for the belief exploration process
     */
    struct Statistics {
        Statistics();
        std::optional<uint64_t> refinementSteps;
        storm::utility::Stopwatch totalTime;

        bool beliefMdpDetectedToBeFinite;
        bool refinementFixpointDetected;

        std::optional<uint64_t> overApproximationStates;
        bool overApproximationBuildAborted;
        storm::utility::Stopwatch overApproximationBuildTime;
        storm::utility::Stopwatch overApproximationCheckTime;
        std::optional<BeliefValueType> overApproximationMaxResolution;

        std::optional<uint64_t> underApproximationStates;
        bool underApproximationBuildAborted;
        storm::utility::Stopwatch underApproximationBuildTime;
        storm::utility::Stopwatch underApproximationCheckTime;
        std::optional<uint64_t> underApproximationStateLimit;
        std::optional<uint64_t> nrClippingAttempts;
        std::optional<uint64_t> nrClippedStates;
        std::optional<uint64_t> nrTruncatedStates;
        storm::utility::Stopwatch clipWatch;
        storm::utility::Stopwatch clippingPreTime;

        bool aborted;
    };

    /**
     * Parameters used for guiding the exploration and abstraction-refinement
     */
    struct HeuristicParameters {
        ValueType gapThreshold;
        ValueType observationThreshold;
        uint64_t sizeThreshold;
        ValueType optimalChoiceValueEpsilon;
    };

    /* Functions */

    /**
     * Returns the pomdp that is to be analyzed
     */
    PomdpModelType const& pomdp() const;

    /**
     * Compute the reachability probability of given target observations on a POMDP using the automatic refinement loop
     *
     * @param targetObservations the set of observations to be reached
     * @param min true if minimum probability is to be computed
     * @return A struct containing the final over-approximation (overApproxValue) and under-approximation (underApproxValue) values
     */
    void refineReachability(storm::Environment const& env, std::set<uint32_t> const& targetObservations, bool min, std::optional<std::string> rewardModelName,
                            storm::pomdp::modelchecker::POMDPValueBounds<ValueType> const& valueBounds, Result& result);
    
    /**
     * Builds and checks an MDP that over-approximates the POMDP behavior, i.e. provides an upper bound for maximizing and a lower bound for minimizing
     * properties
     * @param targetObservations targetObservations the target observations of the objective
     * @param min true if the objective is to minimise the value
     * @param computeRewards true if the objective is to compute a reward value
     * @param refine true if the method is called as part of the abstraction-refinement process
     * @param heuristicParameters parameters used for guiding the exploration
     * @param observationResolutionVector vector of resolutions for each observation used to discretise beliefs
     * @param beliefManager the belief manager to be used
     * @param overApproximation the belief explorer to be used
     * @return True if a fixpoint for the refinement has been detected (i.e. if further refinement steps would not change the MDP)
     */
    bool buildOverApproximation(storm::Environment const& env, std::set<uint32_t> const& targetObservations, bool min, bool computeRewards, bool refine,
                                HeuristicParameters const& heuristicParameters, std::vector<BeliefValueType>& observationResolutionVector,
                                std::shared_ptr<BeliefManagerType>& beliefManager, std::shared_ptr<ExplorerType>& overApproximation);

    /**
     * Builds and checks an MDP that under-approximates the POMDP behavior, i.e. provides a lower bound for maximizing and an upper bound for minimizing
     * properties
     * @param targetObservations targetObservations the target observations of the objective
     * @param min true if the objective is to minimise the value
     * @param computeRewards true if the objective is to compute a reward value
     * @param refine true if the method is called as part of the abstraction-refinement process
     * @param heuristicParameters parameters used for guiding the exploration
     * @param beliefManager the belief manager to be used
     * @param underApproximation the belief explorer to be used
     * @param interactive true if the underapproximation is built as part of an interactive unfolding
     * @return True if a fixpoint for the refinement has been detected (i.e. if further refinement steps would not change the MDP)
     */
    bool buildUnderApproximation(storm::Environment const& env, std::set<uint32_t> const& targetObservations, bool min, bool computeRewards, bool refine,
                                 HeuristicParameters const& heuristicParameters, std::shared_ptr<BeliefManagerType>& beliefManager,
                                 std::shared_ptr<ExplorerType>& underApproximation, bool interactive);

    /**
     * Clips the belief with the given state ID to a belief grid by clipping its direct successor ("grid clipping")
     * Transitions to explored successors and successors on the grid are added, otherwise successors are not generated
     * @param clippingStateId the state ID of the clipping belief
     * @param computeRewards true, if rewards are computed
     * @param min true, if objective is to minimise
     * @param beliefManager the belief manager used
     * @param beliefExplorer the belief MDP explorer used
     */
    void clipToGrid(uint64_t clippingStateId, bool computeRewards, bool min, std::shared_ptr<BeliefManagerType>& beliefManager,
                    std::shared_ptr<ExplorerType>& beliefExplorer);

    /**
     * Clips the belief with the given state ID to a belief grid.
     * If a new candidate is added to the belief space, it is expanded. If necessary, its direct successors are added to the exploration queue to be
     * handled by the main exploration routine.
     * @param clippingStateId the state ID of the clipping belief
     * @param computeRewards true, if rewards are computed
     * @param min true, if objective is to minimise
     * @param beliefManager the belief manager used
     * @param beliefExplorer the belief MDP explorer used
     */
    bool clipToGridExplicitly(uint64_t clippingStateId, bool computeRewards, bool min, std::shared_ptr<BeliefManagerType>& beliefManager,
                              std::shared_ptr<ExplorerType>& beliefExplorer, uint64_t localActionIndex);

    /**
     * Heuristically rates the quality of the approximation described by the given successor observation info.
     * Here, 0 means a bad approximation and 1 means a good approximation.
     */
    BeliefValueType rateObservation(typename ExplorerType::SuccessorObservationInformation const& info, BeliefValueType const& observationResolution,
                                    BeliefValueType const& maxResolution);

    /**
     * Obtains the quality ratings for all observations
     * @param overApproximation pointer to the over-approximation belief explorer
     * @param observationResolutionVector vector containing the resolutions used in the over-approximation for each observation
     * @return vector of ratings
     */
    std::vector<BeliefValueType> getObservationRatings(std::shared_ptr<ExplorerType> const& overApproximation,
                                                       std::vector<BeliefValueType> const& observationResolutionVector);

    /**
     * Obtains the difference between the given lower and upper bounds
     * @param l the lower bound
     * @param u the upper bound
     * @return the difference
     */
    typename PomdpModelType::ValueType getGap(typename PomdpModelType::ValueType const& l, typename PomdpModelType::ValueType const& u);

    /**
     * Sets the command for the interactive belief unfolding
     * @param newUnfoldingControl the new command
     */
    void setUnfoldingControl(UnfoldingControl newUnfoldingControl);

    /* Variables */

    Statistics statistics;
    Options options;

    std::shared_ptr<PomdpModelType> inputPomdp;
    std::shared_ptr<PomdpModelType> preprocessedPomdp;

    storm::utility::ConstantsComparator<BeliefValueType> beliefTypeCC;
    storm::utility::ConstantsComparator<ValueType> valueTypeCC;

    storm::pomdp::modelchecker::POMDPValueBounds<ValueType> pomdpValueBounds;

    std::shared_ptr<ExplorerType> interactiveUnderApproximationExplorer;

    Status unfoldingStatus;
    UnfoldingControl unfoldingControl;
    Result interactiveResult = Result(-storm::utility::infinity<ValueType>(), storm::utility::infinity<ValueType>());
};

}  // namespace modelchecker
}  // namespace pomdp
}  // namespace storm