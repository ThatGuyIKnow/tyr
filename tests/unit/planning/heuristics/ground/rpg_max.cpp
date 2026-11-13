/*
 * Copyright (C) 2025-2026 Dominik Drexler
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "tyr/planning/heuristics/rpg_max.hpp"

#include "../heuristic.hpp"
#include "planning/ground/heuristics/hmax_graph.hpp"
#include "planning/ground/heuristics/rpg.hpp"

namespace tyr::tests
{
namespace
{
class HeuristicFixtureTest : public ::testing::TestWithParam<HeuristicCase>
{
};

struct ReferenceHMax : planning::detail::RPGEvaluator<ReferenceHMax, GroundTag,
    datalog::MinCostAnnotationPolicy<datalog::MaxAggregation>, datalog::TerminationPolicy<datalog::MaxAggregation>>
{
    using Base = planning::detail::RPGEvaluator<ReferenceHMax, GroundTag,
        datalog::MinCostAnnotationPolicy<datalog::MaxAggregation>, datalog::TerminationPolicy<datalog::MaxAggregation>>;
    using Base::Base;
    using Base::evaluate;
    ygg::float_t evaluate(const planning::StateView<GroundTag>& state) { return Base::evaluate(state.get_state_builder()); }
};

TEST(TyrPlanningGroundHMaxGraphTest, FractionalStaticCostsZeroCyclesAndGoalChanges)
{
    const auto domain = std::string(R"((define (domain graph-check)
      (:requirements :strips :numeric-fluents :action-costs)
      (:predicates (start) (middle) (goal)) (:functions (total-cost) (step-cost))
      (:action direct :parameters () :precondition (start)
        :effect (and (not (start)) (goal) (increase (total-cost) 9)))
      (:action cheap :parameters () :precondition (start)
        :effect (and (not (start)) (middle) (increase (total-cost) (step-cost))))
      (:action finish :parameters () :precondition (middle)
        :effect (and (goal) (increase (total-cost) 0)))
      (:action cycle :parameters () :precondition (middle)
        :effect (and (start) (increase (total-cost) 0)))))");
    const auto problem = std::string(R"((define (problem graph-check) (:domain graph-check)
      (:init (start) (= (total-cost) 0) (= (step-cost) 2.5)) (:goal (goal)) (:metric minimize (total-cost))))");
    auto parser = formalism::planning::Parser(domain, "graph-check-domain.pddl");
    auto context = create_heuristic_context<GroundTag>(planning::Task<LiftedTag>::create(parser.parse_task(problem, "graph-check-problem.pddl")));
    ASSERT_TRUE(planning::detail::HMaxGraph::create(context.task, CostMode::GENERAL));
    auto fast = planning::MaxRPGHeuristic<GroundTag>::create(context.task, context.execution_context, CostMode::GENERAL);
    auto reference = ReferenceHMax(context.task, context.execution_context, CostMode::GENERAL);
    const auto initial = context.successor_generator->get_initial_node(*context.state_repository, *context.axiom_evaluator);
    EXPECT_EQ(fast->evaluate(initial.get_state()), 2.5);
    EXPECT_EQ(fast->evaluate(initial.get_state()), reference.evaluate(initial.get_state()));
    const auto successors = context.successor_generator->get_labeled_successor_nodes(initial, *context.state_repository, *context.axiom_evaluator);
    for (const auto& successor : successors) {
        EXPECT_EQ(fast->evaluate(successor.node.get_state()), reference.evaluate(successor.node.get_state()));
        EXPECT_EQ(fast->evaluate(initial.get_state()), 2.5);
    }
    fast->set_goal(context.task->get_task().get_ground_actions().front().get_condition());
    auto worker = fast->make_worker(ygg::ExecutionContext::create(1));
    EXPECT_EQ(worker->evaluate(initial.get_state()), fast->evaluate(initial.get_state()));
}

TEST(TyrPlanningGroundHMaxGraphTest, ReusesBuffersAcrossZeroCostCyclesAndUnreachableGoals)
{
    auto context = create_preferred_action_reset_context<GroundTag>();
    auto graph = std::make_shared<planning::detail::HMaxGraph>();
    graph->consumers.resize(4);
    graph->operators = { { {}, {2}, 9 }, { {}, {0}, 2.5 }, { {0}, {1}, 0 }, { {1}, {0, 2}, 0 } };
    graph->unconditional = { 0, 1 };
    graph->consumers[0] = {2}; graph->consumers[1] = {3};
    auto evaluator = planning::detail::HMaxGraphEvaluator(graph);
    ygg::Builder<planning::State<GroundTag>> empty;
    EXPECT_EQ(evaluator.evaluate(empty, *context.task->get_repository(), {2}), 2.5);
    EXPECT_EQ(evaluator.evaluate(empty, *context.task->get_repository(), {3}), std::numeric_limits<ygg::float_t>::infinity());
    EXPECT_EQ(evaluator.evaluate(empty, *context.task->get_repository(), {0, 2}), 2.5);
    EXPECT_EQ(evaluator.evaluate(empty, *context.task->get_repository(), {}), 0);
}

TEST_P(HeuristicFixtureTest, MatchesGeneralEvaluatorOnSuccessors)
{
    if (std::ranges::none_of(GetParam().configs, [](const auto& config) { return should_check(config.second); })) GTEST_SKIP();
    auto context = create_heuristic_context<GroundTag>(GetParam().domain_file, GetParam().task_file);
    const auto initial = context.successor_generator->get_initial_node(*context.state_repository, *context.axiom_evaluator);
    const auto successors = context.successor_generator->get_labeled_successor_nodes(initial, *context.state_repository, *context.axiom_evaluator);
    for (const auto& [name, expected] : GetParam().configs) {
        if (!should_check(expected)) continue;
        auto fast = planning::MaxRPGHeuristic<GroundTag>::create(context.task, context.execution_context, expected.cost_mode);
        auto reference = ReferenceHMax(context.task, context.execution_context, expected.cost_mode);
        EXPECT_EQ(fast->evaluate(initial.get_state()), reference.evaluate(initial.get_state()));
        size_t count = 0;
        for (const auto& successor : successors) {
            EXPECT_EQ(fast->evaluate(successor.node.get_state()), reference.evaluate(successor.node.get_state()));
            if (++count == 64) break;
        }
        EXPECT_EQ(fast->evaluate(initial.get_state()), reference.evaluate(initial.get_state()));
    }
}

TEST(TyrPlanningHeuristicWorkerTest, SnapshotsConfiguredGoal) { expect_worker_snapshots_configured_goal<GroundTag, tyr::planning::MaxRPGHeuristic>(); }

TEST(TyrPlanningHeuristicEvaluationTest, AlternatingStatesReplacePreviousModel)
{ expect_alternating_states_replace_previous_model<GroundTag, tyr::planning::MaxRPGHeuristic>(); }

TEST_P(HeuristicFixtureTest, InitialStateMatchesFixture) { check_heuristic_case<GroundTag, tyr::planning::MaxRPGHeuristic>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(TyrPlanningHeuristicFixture,
                         HeuristicFixtureTest,
                         ::testing::ValuesIn(load_cases("tests/fixtures/planning/heuristics/ground/rpg_max.json")),
                         [](const testing::TestParamInfo<HeuristicCase>& param_info) { return param_info.param.name; });
}
}
