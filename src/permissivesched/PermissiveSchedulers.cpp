
#include "PermissiveSchedulers.h"

#include "src/models/sparse/StandardRewardModel.h"
#include "../utility/solver.h"
#include "../utility/graph.h"
#include "../modelchecker/propositional/SparsePropositionalModelChecker.h"
#include "../modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "MILPPermissiveSchedulers.h"
#include "SmtBasedPermissiveSchedulers.h"
#include "src/exceptions/NotImplementedException.h"
#include "src/utility/macros.h"

namespace storm {
    namespace ps {
        
        template<typename RM>
        boost::optional<SubMDPPermissiveScheduler<RM>> computePermissiveSchedulerViaMILP(storm::models::sparse::Mdp<double, RM> const& mdp, storm::logic::ProbabilityOperatorFormula const& safeProp) {
            storm::modelchecker::SparsePropositionalModelChecker<storm::models::sparse::Mdp<double, RM>> propMC(mdp);
            assert(safeProp.getSubformula().isEventuallyFormula());
            auto backwardTransitions = mdp.getBackwardTransitions();
            storm::storage::BitVector goalstates = propMC.check(safeProp.getSubformula().asEventuallyFormula().getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();
            goalstates = storm::utility::graph::performProb1A(mdp, backwardTransitions, storm::storage::BitVector(goalstates.size(), true), goalstates);
            storm::storage::BitVector sinkstates = storm::utility::graph::performProb0A(mdp,backwardTransitions, storm::storage::BitVector(goalstates.size(), true), goalstates);
            
            auto solver = storm::utility::solver::getLpSolver("Gurobi", storm::solver::LpSolverTypeSelection::Gurobi);
            MilpPermissiveSchedulerComputation<storm::models::sparse::StandardRewardModel<double>> comp(*solver, mdp, goalstates, sinkstates);
            STORM_LOG_THROW(!storm::logic::isStrict(safeProp.getComparisonType()), storm::exceptions::NotImplementedException, "Strict bounds are not supported");
            comp.calculatePermissiveScheduler(storm::logic::isLowerBound(safeProp.getComparisonType()), safeProp.getBound());
            //comp.dumpLpToFile("milpdump.lp");
            std::cout << "Found Solution: " << (comp.foundSolution() ? "yes" : "no") << std::endl;
            if(comp.foundSolution()) {
                return boost::optional<SubMDPPermissiveScheduler<RM>>(comp.getScheduler());
            } else {
                return boost::optional<SubMDPPermissiveScheduler<RM>>();
            }
        }
        
        template<typename RM>
        boost::optional<SubMDPPermissiveScheduler<RM>> computePermissiveSchedulerViaMC(std::shared_ptr<storm::models::sparse::Mdp<double, RM>> mdp, storm::logic::ProbabilityOperatorFormula const& safeProp) {
            
        }
        
        template<typename RM>
        boost::optional<SubMDPPermissiveScheduler<RM>> computePermissiveSchedulerViaSMT(storm::models::sparse::Mdp<double, RM> const& mdp, storm::logic::ProbabilityOperatorFormula const& safeProp) {
            storm::modelchecker::SparsePropositionalModelChecker<storm::models::sparse::Mdp<double, RM>> propMC(mdp);
            assert(safeProp.getSubformula().isEventuallyFormula());
            auto backwardTransitions = mdp.getBackwardTransitions();
            storm::storage::BitVector goalstates = propMC.check(safeProp.getSubformula().asEventuallyFormula().getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();
            goalstates = storm::utility::graph::performProb1A(mdp, backwardTransitions, storm::storage::BitVector(goalstates.size(), true), goalstates);
            storm::storage::BitVector sinkstates = storm::utility::graph::performProb0A(mdp, backwardTransitions, storm::storage::BitVector(goalstates.size(), true), goalstates);
            
            std::shared_ptr<storm::expressions::ExpressionManager> expressionManager = std::make_shared<storm::expressions::ExpressionManager>();
            auto solver = storm::utility::solver::getSmtSolver(*expressionManager);
            SmtPermissiveSchedulerComputation<storm::models::sparse::StandardRewardModel<double>> comp(*solver, mdp, goalstates, sinkstates);
            STORM_LOG_THROW(!storm::logic::isStrict(safeProp.getComparisonType()), storm::exceptions::NotImplementedException, "Strict bounds are not supported");
            comp.calculatePermissiveScheduler(storm::logic::isLowerBound(safeProp.getComparisonType()), safeProp.getBound());
            if(comp.foundSolution()) {
                return boost::optional<SubMDPPermissiveScheduler<RM>>(comp.getScheduler());
            } else {
                return boost::optional<SubMDPPermissiveScheduler<RM>>();
            }
            return boost::none;
        }
        
        template boost::optional<SubMDPPermissiveScheduler<>> computePermissiveSchedulerViaMILP(storm::models::sparse::Mdp<double> const& mdp, storm::logic::ProbabilityOperatorFormula const& safeProp);
        template boost::optional<SubMDPPermissiveScheduler<>> computePermissiveSchedulerViaSMT(storm::models::sparse::Mdp<double> const& mdp, storm::logic::ProbabilityOperatorFormula const& safeProp);
        
        
    }
}
