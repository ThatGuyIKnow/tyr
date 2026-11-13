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

#include "../ground/heuristics/rpg.hpp"
#include "../ground/heuristics/hmax_graph.hpp"
#include "../lifted/heuristics/rpg.hpp"
#include "rpg.hpp"
#include "tyr/datalog/policies/annotation.hpp"
#include "tyr/datalog/policies/cost.hpp"
#include "tyr/datalog/policies/termination.hpp"

#include <utility>

namespace tyr::planning
{

template<TaskKind Kind>
struct MaxRPGHeuristic<Kind>::Impl :
    detail::RPGEvaluator<Impl, Kind, datalog::MinCostAnnotationPolicy<datalog::MaxAggregation>, datalog::TerminationPolicy<datalog::MaxAggregation>>
{
    using Base =
        detail::RPGEvaluator<Impl, Kind, datalog::MinCostAnnotationPolicy<datalog::MaxAggregation>, datalog::TerminationPolicy<datalog::MaxAggregation>>;

    using Base::Base;
};

template<>
struct MaxRPGHeuristic<GroundTag>::Impl
{
    struct General : detail::RPGEvaluator<General, GroundTag, datalog::MinCostAnnotationPolicy<datalog::MaxAggregation>,
                                         datalog::TerminationPolicy<datalog::MaxAggregation>>
    {
        using Base = detail::RPGEvaluator<General, GroundTag, datalog::MinCostAnnotationPolicy<datalog::MaxAggregation>,
                                          datalog::TerminationPolicy<datalog::MaxAggregation>>;
        using Base::Base;
    };

    TaskPtr<GroundTag> task;
    ygg::ExecutionContextPtr execution_context;
    CostMode cost_mode;
    std::shared_ptr<const detail::HMaxGraph> graph;
    std::optional<detail::HMaxGraphEvaluator> fast;
    std::vector<ygg::uint_t> goals;
    std::unique_ptr<General> general;

    Impl(TaskPtr<GroundTag> task_, ygg::ExecutionContextPtr context, CostMode mode) :
        task(std::move(task_)), execution_context(std::move(context)), cost_mode(mode), graph(detail::HMaxGraph::create(task, mode))
    {
        if (graph) {
            fast.emplace(graph);
            goals = detail::HMaxGraph::atoms(task->get_task().get_goal());
        } else general = std::make_unique<General>(task, execution_context, cost_mode);
    }

    Impl(const Impl& other, ygg::ExecutionContextPtr context) :
        task(other.task), execution_context(std::move(context)), cost_mode(other.cost_mode), graph(other.graph), goals(other.goals)
    {
        if (other.general) general = std::make_unique<General>(*other.general, execution_context);
        else fast.emplace(graph);
    }

    void set_goal(formalism::planning::ConjunctiveConditionView<GroundTag> goal)
    {
        if (!general && detail::HMaxGraph::supports(goal)) goals = detail::HMaxGraph::atoms(goal);
        else {
            if (!general) {
                general = std::make_unique<General>(task, execution_context, cost_mode);
                fast.reset();
            }
            general->set_goal(goal);
        }
    }

    ygg::float_t evaluate(const ygg::Builder<State<GroundTag>>& state)
    { return general ? general->evaluate(state) : fast->evaluate(state, *task->get_repository(), goals); }

    void print_summary(size_t verbosity) const { if (general) general->print_summary(verbosity); }
};

template<TaskKind Kind>
MaxRPGHeuristic<Kind>::MaxRPGHeuristic(TaskPtr<Kind> task, ygg::ExecutionContextPtr execution_context, CostMode cost_mode) :
    m_impl(std::make_unique<Impl>(std::move(task), std::move(execution_context), cost_mode))
{
}

template<TaskKind Kind>
MaxRPGHeuristic<Kind>::MaxRPGHeuristic(std::unique_ptr<Impl> impl) : m_impl(std::move(impl))
{
}

template<TaskKind Kind>
MaxRPGHeuristic<Kind>::~MaxRPGHeuristic() = default;

template<TaskKind Kind>
MaxRPGHeuristic<Kind>::MaxRPGHeuristic(MaxRPGHeuristic&&) noexcept = default;

template<TaskKind Kind>
MaxRPGHeuristic<Kind>& MaxRPGHeuristic<Kind>::operator=(MaxRPGHeuristic&&) noexcept = default;

template<TaskKind Kind>
MaxRPGHeuristicPtr<Kind> MaxRPGHeuristic<Kind>::create(TaskPtr<Kind> task, ygg::ExecutionContextPtr execution_context, CostMode cost_mode)
{ return std::make_shared<MaxRPGHeuristic<Kind>>(std::move(task), std::move(execution_context), cost_mode); }

template<TaskKind Kind>
void MaxRPGHeuristic<Kind>::set_goal(formalism::planning::ConjunctiveConditionView<GroundTag> goal)
{ m_impl->set_goal(goal); }

template<TaskKind Kind>
ygg::float_t MaxRPGHeuristic<Kind>::evaluate(const ygg::Builder<State<Kind>>& state)
{ return m_impl->evaluate(state); }

template<TaskKind Kind>
HeuristicPtr<Kind> MaxRPGHeuristic<Kind>::make_worker(ygg::ExecutionContextPtr execution_context) const
{ return HeuristicPtr<Kind>(new MaxRPGHeuristic(std::make_unique<Impl>(*m_impl, std::move(execution_context)))); }

template<TaskKind Kind>
void MaxRPGHeuristic<Kind>::print_summary(size_t verbosity) const
{ m_impl->print_summary(verbosity); }

template class MaxRPGHeuristic<GroundTag>;
template class MaxRPGHeuristic<LiftedTag>;

}
