/* Copyright (C) 2026 Dominik Drexler. SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef TYR_SRC_PLANNING_GROUND_HEURISTICS_HMAX_GRAPH_HPP_
#define TYR_SRC_PLANNING_GROUND_HEURISTICS_HMAX_GRAPH_HPP_

#include "tyr/formalism/planning/views.hpp"
#include "tyr/planning/applicability.hpp"
#include "tyr/planning/ground/task.hpp"
#include "tyr/planning/state_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tyr::planning::detail
{
struct HMaxGraph
{
    using Atom = formalism::planning::Atom<GroundTag, formalism::FluentTag>;
    using Id = ygg::uint_t;
    struct Operator
    {
        std::vector<Id> preconditions;
        std::vector<Id> effects;
        ygg::float_t cost;
    };
    std::vector<Operator> operators;
    std::vector<std::vector<Id>> consumers;
    std::vector<Id> unconditional;

    static bool supports(formalism::planning::ConjunctiveConditionView<GroundTag> condition)
    { return condition.get_numeric_constraints().empty() && condition.get_literals<formalism::DerivedTag>().empty(); }

    static std::vector<Id> atoms(formalism::planning::ConjunctiveConditionView<GroundTag> condition)
    {
        std::vector<Id> result;
        for (const auto fact : condition.get_facts<formalism::PositiveTag>())
            if (const auto atom = fact.get_atom())
                result.push_back(ygg::uint_t(atom->get_index()));
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    static bool static_expression(formalism::planning::FunctionExpressionView<GroundTag> expression)
    {
        return ygg::visit(
            [](auto value) -> bool
            {
                using V = decltype(value);
                if constexpr (std::same_as<V, ygg::float_t>
                              || ygg::is_view_of_v<V, ygg::Index<formalism::planning::FunctionTerm<GroundTag, formalism::StaticTag>>>)
                    return true;
                else if constexpr (ygg::is_view_of_v<V, ygg::Data<formalism::planning::ArithmeticOperator<GroundTag>>>)
                    return ygg::visit(
                        [](auto operation)
                        {
                            if constexpr (requires { operation.get_arg(); })
                                return static_expression(operation.get_arg());
                            else if constexpr (requires { operation.get_lhs(); })
                                return static_expression(operation.get_lhs()) && static_expression(operation.get_rhs());
                            else
                            {
                                for (const auto arg : operation.get_args())
                                    if (!static_expression(arg))
                                        return false;
                                return true;
                            }
                        },
                        value.get_variant());
                else
                    return false;
            },
            expression.get_variant());
    }

    // A missing cost requests the general RPG evaluator for the whole task.
    static std::optional<ygg::float_t>
    relaxed_action_cost(formalism::planning::ActionView<GroundTag> action, CostMode mode, bool uses_metric, const StateContext<GroundTag>& state)
    {
        auto cost = mode == CostMode::UNIT ? ygg::float_t(1) : ygg::float_t(0);
        for (const auto effect : action.get_effects())
        {
            if (!supports(effect.get_condition()) || !effect.get_effect().get_numeric_effects().empty())
                return {};
            if (uses_metric)
                if (const auto auxiliary = effect.get_effect().get_auxiliary_numeric_effect())
                {
                    // Conditional metric effects retain the general evaluator's validation.
                    const auto condition = effect.get_condition();
                    if (!condition.get_literals<formalism::StaticTag>().empty() || !condition.get_facts<formalism::PositiveTag>().empty()
                        || !condition.get_facts<formalism::NegativeTag>().empty())
                        return {};
                    const auto increment = ygg::visit(
                        [&](auto numeric) -> std::optional<ygg::float_t>
                        {
                            if (numeric.get_operator() != formalism::NumericEffectOperatorKind::Increase || !static_expression(numeric.get_fexpr()))
                                return std::nullopt;
                            const auto amount = evaluate(numeric.get_fexpr(), state);
                            if (!std::isfinite(amount) || amount < 0)
                                return std::nullopt;
                            return amount;
                        },
                        auxiliary.value().get_variant());
                    if (!increment)
                        return {};
                    cost += *increment;
                }
        }
        if (!std::isfinite(cost))
            return {};
        return cost;
    }

    static std::shared_ptr<const HMaxGraph> create(const TaskPtr<GroundTag>& task, CostMode mode)
    {
        const auto view = task->get_task();
        if (!supports(view.get_goal()))
            return {};
        if (mode == CostMode::GENERAL && view.get_metric())
        {
            const auto auxiliary = view.get_auxiliary_fterm_value();
            if (!auxiliary
                || !ygg::visit(
                    [&](auto value)
                    {
                        using V = decltype(value);
                        if constexpr (ygg::is_view_of_v<V, ygg::Index<formalism::planning::FunctionTerm<GroundTag, formalism::AuxiliaryTag>>>)
                            return value == auxiliary.value().get_fterm();
                        else
                            return false;
                    },
                    view.get_metric().value().get_fexpr().get_variant()))
                return {};
        }
        auto graph = std::make_shared<HMaxGraph>();
        graph->consumers.resize(task->get_repository()->size<Atom>());
        ygg::Builder<State<GroundTag>> empty;
        const auto state = StateContext<GroundTag>(*task, empty, 0);
        for (const auto action : view.get_ground_actions())
        {
            if (!supports(action.get_condition()))
                return {};
            if (!is_statically_applicable(action, task->get_static_state()))
                continue;
            const auto cost = relaxed_action_cost(action, mode, mode == CostMode::GENERAL && view.get_metric(), state);
            if (!cost)
                return {};
            const auto action_preconditions = atoms(action.get_condition());
            for (const auto conditional : action.get_effects())
            {
                if (!is_statically_applicable(conditional.get_condition(), task->get_static_state()))
                    continue;
                Operator op { action_preconditions, {}, *cost };
                const auto extra = atoms(conditional.get_condition());
                op.preconditions.insert(op.preconditions.end(), extra.begin(), extra.end());
                std::sort(op.preconditions.begin(), op.preconditions.end());
                op.preconditions.erase(std::unique(op.preconditions.begin(), op.preconditions.end()), op.preconditions.end());
                for (const auto fact : conditional.get_effect().get_facts<formalism::PositiveTag>())
                    if (const auto atom = fact.get_atom())
                        op.effects.push_back(ygg::uint_t(atom->get_index()));
                if (op.effects.empty())
                    continue;
                const auto id = static_cast<Id>(graph->operators.size());
                if (op.preconditions.empty())
                    graph->unconditional.push_back(id);
                for (const auto atom : op.preconditions)
                    graph->consumers.at(atom).push_back(id);
                graph->operators.push_back(std::move(op));
            }
        }
        return graph;
    }
};

class HMaxGraphEvaluator
{
    using Id = HMaxGraph::Id;
    using Cost = ygg::float_t;
    struct FactLabel
    {
        uint64_t generation = 0;
        Cost cost = 0;
    };
    struct PendingOperator
    {
        uint64_t generation = 0;
        size_t remaining = 0;
    };
    std::shared_ptr<const HMaxGraph> m_graph;
    std::vector<FactLabel> m_facts;
    std::vector<PendingOperator> m_operators;
    std::vector<std::pair<Cost, Id>> m_queue;
    uint64_t m_generation = 0;

    // A fresh generation makes old fact labels and precondition counts logically empty.
    void start_evaluation()
    {
        if (++m_generation == 0)
        {
            for (auto& fact : m_facts)
                fact.generation = 0;
            for (auto& op : m_operators)
                op.generation = 0;
            m_generation = 1;
        }
        m_queue.clear();
    }

    void relax_fact(Id atom, Cost cost)
    {
        auto& fact = m_facts.at(atom);
        if (fact.generation == m_generation && fact.cost <= cost)
            return;
        fact = { m_generation, cost };
        m_queue.emplace_back(cost, atom);
        std::push_heap(m_queue.begin(), m_queue.end(), std::greater<>());
    }
    void fire(Id id, Cost max_precondition_cost)
    {
        const auto& op = m_graph->operators[id];
        const auto cost = max_precondition_cost + op.cost;
        if (!std::isfinite(cost))
            return;
        for (const auto atom : op.effects)
            relax_fact(atom, cost);
    }

public:
    explicit HMaxGraphEvaluator(std::shared_ptr<const HMaxGraph> graph) :
        m_graph(std::move(graph)),
        m_facts(m_graph->consumers.size()),
        m_operators(m_graph->operators.size())
    {
    }

    Cost evaluate(const ygg::Builder<State<GroundTag>>& state, const formalism::planning::Repository& repository, const std::vector<Id>& goals)
    {
        if (goals.empty())
            return 0;
        start_evaluation();
        for (const auto atom : state.get_atoms_view<formalism::FluentTag>(repository))
            relax_fact(ygg::uint_t(atom.get_index()), 0);
        for (const auto id : m_graph->unconditional)
            fire(id, 0);
        // Goals are distinct. The last goal settled has the maximum goal cost.
        auto remaining_goals = goals.size();
        while (!m_queue.empty())
        {
            std::pop_heap(m_queue.begin(), m_queue.end(), std::greater<>());
            const auto [cost, atom] = m_queue.back();
            m_queue.pop_back();
            // A cheaper label may have superseded this queue entry.
            if (cost != m_facts[atom].cost)
                continue;
            if (std::binary_search(goals.begin(), goals.end(), atom) && --remaining_goals == 0)
                return cost;
            for (const auto id : m_graph->consumers[atom])
            {
                auto& pending = m_operators[id];
                if (pending.generation != m_generation)
                    pending = { m_generation, m_graph->operators[id].preconditions.size() };
                // Facts settle in increasing cost order, so the last precondition supplies the maximum.
                if (--pending.remaining == 0)
                    fire(id, cost);
            }
        }
        return std::numeric_limits<Cost>::infinity();
    }
};
}
#endif
