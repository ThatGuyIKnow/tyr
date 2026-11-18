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

#ifndef TYR_SRC_DATALOG_GROUND_SCHEDULER_IMPL_HPP_
#define TYR_SRC_DATALOG_GROUND_SCHEDULER_IMPL_HPP_

#include "tyr/datalog/applicability.hpp"
#include "tyr/datalog/fact_sets.hpp"
#include "tyr/datalog/ground/contexts/program.hpp"
#include "tyr/datalog/ground/rule_instance.hpp"
#include "tyr/datalog/ground/scheduler.hpp"
#include "tyr/datalog/rule_evaluation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>
#include <yggdrasil/core/config.hpp>

namespace tyr::datalog
{
template<formalism::RelationKind R, AnnotationPolicyConcept AP, TerminationPolicyConcept TP, RuleCostPolicyConcept CP>
void Scheduler<GroundTag>::schedule(formalism::datalog::RuleView<GroundTag, R> rule, ProgramExecutionContext<GroundTag, AP, TP, CP>& ctx)
{
    if (get_states<R>()[ygg::uint_t(rule.get_index())].unsatisfied_count != 0)
        return;

    auto instance = RuleInstance<GroundTag, R>(rule);
    auto selector = ctx.out().numeric_support_selector();
    auto& workspace = m_scratch.rule_evaluation;
    const auto input = RuleEvaluationInput { selector, ctx.out().annotations() };
    const auto priority = evaluate_rule_priority(instance, ctx.out().annotation_policy(), ctx.out().cost_policy(), input, workspace);
    if (priority)
        enqueue(rule, *priority);
}

template<formalism::RelationKind R, AnnotationPolicyConcept AP, TerminationPolicyConcept TP, RuleCostPolicyConcept CP>
void Scheduler<GroundTag>::update_numeric_constraint_satisfaction(formalism::datalog::RuleView<GroundTag, R> rule,
                                                                  ProgramExecutionContext<GroundTag, AP, TP, CP>& ctx)
{
    const auto rule_index = rule.get_index();
    auto& states = get_states<R>();
    auto& satisfied = states[ygg::uint_t(rule_index)].numeric_constraint_satisfied;
    const auto numeric_constraints = rule.get_body().get_numeric_constraints();
    if (satisfied.size() != numeric_constraints.size())
        satisfied.assign(numeric_constraints.size(), false);

    const auto fact_sets = FactSets { ctx.in().facts().fact_sets, ctx.out().facts().fact_sets };
    for (ygg::uint_t i = 0; i < numeric_constraints.size(); ++i)
    {
        if (satisfied[i] || !evaluate(numeric_constraints[i], fact_sets))
            continue;

        satisfied[i] = true;
        auto& unsatisfied_count = states[ygg::uint_t(rule_index)].unsatisfied_count;
        assert(unsatisfied_count > 0);
        --unsatisfied_count;
    }
}

template<formalism::RelationKind R, AnnotationPolicyConcept AP, TerminationPolicyConcept TP, RuleCostPolicyConcept CP>
void Scheduler<GroundTag>::notify_predicate_generated_for(formalism::datalog::PredicateBindingIndex<formalism::FluentTag> fact,
                                                          ProgramExecutionContext<GroundTag, AP, TP, CP>& ctx)
{
    const auto* dependent_rules = ctx.in().template dependencies<R>().fluent_precondition_to_rules.find(fact);
    if (!dependent_rules)
        return;

    for (const auto dependent_rule : *dependent_rules)
    {
        auto& unsatisfied_count = get_states<R>()[ygg::uint_t(dependent_rule.get_index())].unsatisfied_count;
        if (unsatisfied_count == 0)
            continue;

        if constexpr (std::same_as<R, formalism::PredicateTag>)
            if (m_fires_once) {
                auto& support = m_support[ygg::uint_t(dependent_rule.get_index())];
                support = std::max(support, ctx.out().annotations().fetch_cost(fact));
                if (--unsatisfied_count == 0) m_ready.push_back(dependent_rule);
                continue;
            }

        --unsatisfied_count;
        schedule(dependent_rule, ctx);
    }
}

template<formalism::RelationKind R, AnnotationPolicyConcept AP, TerminationPolicyConcept TP, RuleCostPolicyConcept CP>
void Scheduler<GroundTag>::notify_numeric_changed_for(formalism::datalog::FunctionBindingIndex<formalism::FluentTag> term,
                                                      ProgramExecutionContext<GroundTag, AP, TP, CP>& ctx)
{
    const auto* dependent_rules = ctx.in().template dependencies<R>().fluent_function_term_to_rules.find(term);
    if (!dependent_rules)
        return;

    for (const auto dependent_rule : *dependent_rules)
    {
        update_numeric_constraint_satisfaction(dependent_rule, ctx);
        if (get_states<R>()[ygg::uint_t(dependent_rule.get_index())].unsatisfied_count == 0)
            schedule(dependent_rule, ctx);
    }
}

template<AnnotationPolicyConcept AP, TerminationPolicyConcept TP, RuleCostPolicyConcept CP>
void Scheduler<GroundTag>::begin_stratum(ProgramExecutionContext<GroundTag, AP, TP, CP>& ctx)
{
    // A resumed solve may need to restore staged annotations to the frontier.
    if (m_fires_once && !m_new_run) {
        const auto facts = FactSets {ctx.in().facts().fact_sets, ctx.out().facts().fact_sets};
        initialize_rule_states<formalism::PredicateTag>(facts);
    }
    m_fires_once = false;
    m_ready.clear();
    const auto new_run = std::exchange(m_new_run, false);
    // Dynamic numeric support and cost overrides retain the general evaluator.
    if constexpr (std::same_as<AP, MinCostAnnotationPolicy<MaxAggregation>> && std::same_as<CP, RuleCostPolicy>) {
        if (!m_checked_once) {
            m_checked_once = true;
            m_once_supported = m_program.get_rules<formalism::FunctionTag>().empty();
            const auto facts = FactSets {ctx.in().facts().fact_sets, ctx.out().facts().fact_sets};
            for (const auto rule : m_program.get_rules<formalism::PredicateTag>()) {
                auto edge = static_cast<Cost>(rule.get_cost());
                m_once_supported &= rule.get_body().get_numeric_constraints().empty()
                    && formalism::datalog::collect_fluent_reads(rule).empty();
                for (const auto literal : rule.get_body().template get_literals<formalism::FluentTag>())
                    m_once_supported &= literal.get_polarity();
                for (const auto effect : rule.get_metric_effects())
                    ygg::visit([&](auto numeric) {
                        if (!ctx.out().cost_policy().is_metric_target(numeric.get_fterm().get_row().get_index())) return;
                        if (numeric.get_operator() != formalism::NumericEffectOperatorKind::Increase) {
                            m_once_supported = false;
                            return;
                        }
                        if (!formalism::datalog::collect_fluent_reads(rule).empty()) return;
                        const auto value = evaluate(numeric.get_fexpr(), facts);
                        if (empty(value) || !std::isfinite(lower(value)) || lower(value) != upper(value)) m_once_supported = false;
                        else edge += std::max(lower(value), Cost(0));
                    }, effect.get_variant());
                m_once_supported &= std::isfinite(edge) && edge >= 0;
                m_edges.push_back(edge);
                m_conditions.push_back(rule.get_body().template get_literals<formalism::FluentTag>().size());
            }
        }
        if (m_once_supported && new_run) {
            m_fires_once = true;
            m_support.assign(m_edges.size(), Cost(0));
            for (const auto rule : m_program.get_rules<formalism::PredicateTag>()) {
                const auto i = ygg::uint_t(rule.get_index());
                get_states<formalism::PredicateTag>()[i].unsatisfied_count = m_conditions[i];
                if (m_conditions[i] == 0)
                    m_ready.push_back(rule);
            }
            for (const auto& set : ctx.out().facts().fact_sets.predicate.get_sets())
                for (const auto binding : set.get_bindings())
                    notify_predicate_generated_for<formalism::PredicateTag>(binding.get_index(), ctx);
            return;
        }
    }
    for (const auto rule : m_program.get_rules<formalism::PredicateTag>())
        update_numeric_constraint_satisfaction(rule, ctx);
    for (const auto rule : m_program.get_rules<formalism::FunctionTag>())
        update_numeric_constraint_satisfaction(rule, ctx);

    for (const auto rule : m_program.get_rules<formalism::PredicateTag>())
        schedule(rule, ctx);
    for (const auto rule : m_program.get_rules<formalism::FunctionTag>())
        schedule(rule, ctx);
}

template<AnnotationPolicyConcept AP, TerminationPolicyConcept TP, RuleCostPolicyConcept CP>
void Scheduler<GroundTag>::notify_predicate_generated(formalism::datalog::PredicateBindingIndex<formalism::FluentTag> fact,
                                                      ProgramExecutionContext<GroundTag, AP, TP, CP>& ctx)
{
    ++m_statistics.num_facts_derived;
    notify_predicate_generated_for<formalism::PredicateTag>(fact, ctx);
    notify_predicate_generated_for<formalism::FunctionTag>(fact, ctx);
}

template<AnnotationPolicyConcept AP, TerminationPolicyConcept TP, RuleCostPolicyConcept CP>
void Scheduler<GroundTag>::notify_numeric_changed(formalism::datalog::FunctionBindingIndex<formalism::FluentTag> term,
                                                  ProgramExecutionContext<GroundTag, AP, TP, CP>& ctx)
{
    notify_numeric_changed_for<formalism::PredicateTag>(term, ctx);
    notify_numeric_changed_for<formalism::FunctionTag>(term, ctx);
}

template<AnnotationPolicyConcept AP, TerminationPolicyConcept TP, RuleCostPolicyConcept CP>
void Scheduler<GroundTag>::notify_generated(const CostBuckets::Bucket& bucket, ProgramExecutionContext<GroundTag, AP, TP, CP>& ctx)
{
    if (m_fires_once) {
        for (const auto fact : bucket.predicate) notify_predicate_generated(fact, ctx);
        return;
    }
    m_scratch.changed_facts.assign(bucket.predicate.begin(), bucket.predicate.end());
    m_scratch.changed_terms.clear();
    m_scratch.changed_terms.reserve(bucket.function.size());
    for (const auto& entry : bucket.function)
        m_scratch.changed_terms.push_back(entry.first);

    std::sort(m_scratch.changed_facts.begin(), m_scratch.changed_facts.end());
    std::sort(m_scratch.changed_terms.begin(), m_scratch.changed_terms.end());
    for (const auto fact : m_scratch.changed_facts)
        notify_predicate_generated(fact, ctx);
    for (const auto term : m_scratch.changed_terms)
        notify_numeric_changed(term, ctx);
}

}

#endif
