#include "test/storm_gtest.h"
#include "storm-config.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm-parsers/parser/PrismParser.h"
#include "storm/builder/ExplicitModelBuilder.h"
#include "storm/api/storm.h"
#include "storm-parsers/api/storm-parsers.h"
#include "storm-pomdp/analysis/FormulaInformation.h"
#include "storm-pomdp/analysis/QualitativeAnalysisOnGraphs.h"
#include "storm-pomdp/analysis/CanonicityChecker.h"

TEST(CanonicityChecking, Canonic) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/pomdp/canonic.prism");
    std::shared_ptr<storm::logic::Formula const> formula = storm::api::parsePropertiesForPrismProgram("Pmax=? [F \"goal\" ]", program).front().getRawFormula();
    std::shared_ptr<storm::models::sparse::Pomdp<double>> pomdp = storm::api::buildSparseModel<double>(program, {formula})->as<storm::models::sparse::Pomdp<double>>();
    auto canonicityChecker = storm::pomdp::analysis::CanonicityChecker<double>();
    EXPECT_TRUE(canonicityChecker.check(pomdp));
}

TEST(CanonicityChecking, NonCanonic1) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/pomdp/nonCanonic1.prism");
    std::shared_ptr<storm::logic::Formula const> formula = storm::api::parsePropertiesForPrismProgram("Pmax=? [F \"goal\" ]", program).front().getRawFormula();
    std::shared_ptr<storm::models::sparse::Pomdp<double>> pomdp = storm::api::buildSparseModel<double>(program, {formula})->as<storm::models::sparse::Pomdp<double>>();
    auto canonicityChecker = storm::pomdp::analysis::CanonicityChecker<double>();
    EXPECT_FALSE(canonicityChecker.check(pomdp));
}

TEST(CanonicityChecking, NonCanonic2) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/pomdp/nonCanonic2.prism");
    std::shared_ptr<storm::logic::Formula const> formula = storm::api::parsePropertiesForPrismProgram("Pmax=? [F \"goal\" ]", program).front().getRawFormula();
    std::shared_ptr<storm::models::sparse::Pomdp<double>> pomdp = storm::api::buildSparseModel<double>(program, {formula})->as<storm::models::sparse::Pomdp<double>>();
    auto canonicityChecker = storm::pomdp::analysis::CanonicityChecker<double>();
    EXPECT_FALSE(canonicityChecker.check(pomdp));
}

TEST(CanonicityChecking, NonCanonic3) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/pomdp/nonCanonic3.prism");
    std::shared_ptr<storm::logic::Formula const> formula = storm::api::parsePropertiesForPrismProgram("Pmax=? [F \"goal\" ]", program).front().getRawFormula();
    std::shared_ptr<storm::models::sparse::Pomdp<double>> pomdp = storm::api::buildSparseModel<double>(program, {formula})->as<storm::models::sparse::Pomdp<double>>();
    auto canonicityChecker = storm::pomdp::analysis::CanonicityChecker<double>();
    EXPECT_FALSE(canonicityChecker.check(pomdp));
}