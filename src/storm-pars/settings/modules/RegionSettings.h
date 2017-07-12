#pragma once

#include "storm-pars/modelchecker/region/RegionCheckEngine.h"
#include "storm-pars/modelchecker/region/RegionResultHypothesis.h"

#include "storm/settings/modules/ModuleSettings.h"

namespace storm {
    namespace settings {
        namespace modules {

            /*!
             * This class represents the settings for parametric model checking.
             */
            class RegionSettings : public ModuleSettings {
            public:
                
                /*!
                 * Creates a new set of parametric model checking settings.
                 */
                RegionSettings();
				
				/*!
				 * Retrieves whether region(s) were declared
				 */
				bool isRegionSet() const;
				
				/*!
				 * Retrieves the region definition string
				 */
				std::string getRegionString() const;
				
				/*!
				 * Retrieves whether region(s) were declared
				 */
				bool isHypothesisSet() const;
				
				/*!
				 * Retrieves the region definition string
				 */
				storm::modelchecker::RegionResultHypothesis getHypothesis() const;
				
				/*!
				 * Retrieves whether region refinement is enabled
				 */
				bool isRefineSet() const;
				
				/*!
				 * Retrieves the threshold considered for iterative region refinement.
				 * The refinement converges as soon as the fraction of unknown area falls below this threshold
				 */
				double getCoverageThreshold() const;
				
                /*!
                 * Retrieves whether a depth threshold has been set for refinement
                 */
                bool isDepthLimitSet() const;
                
                /*!
                 * Returns the depth threshold (if set). It is illegal to call this method if no depth threshold has been set.
                 */
                uint64_t getDepthLimit() const;
                
				/*!
				 * Retrieves which type of region check should be performed
				 */
				storm::modelchecker::RegionCheckEngine getRegionCheckEngine() const;
				
				/*!
				 * Retrieves whether no illustration of the result should be printed.
				 */
                bool isPrintNoIllustrationSet() const;
                
                /*!
                 * Retrieves whether the full result should be printed
                 */
                bool isPrintFullResultSet() const;
                
                const static std::string moduleName;
                
            private:
				const static std::string regionOptionName;
				const static std::string regionShortOptionName;
				const static std::string hypothesisOptionName;
				const static std::string hypothesisShortOptionName;
				const static std::string refineOptionName;
				const static std::string checkEngineOptionName;
				const static std::string printNoIllustrationOptionName;
				const static std::string printFullResultOptionName;
            };
            
        } // namespace modules
    } // namespace settings
} // namespace storm

