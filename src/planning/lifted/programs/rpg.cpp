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

#include "tyr/planning/programs/rpg.hpp"

#include "../../programs/common.hpp"
#include "../../programs/translated_program.hpp"
#include "tyr/datalog/rule_splitting.hpp"
#include "tyr/datalog/static_literal_decomposition.hpp"
#include "tyr/formalism/datalog/copy.hpp"
#include "tyr/formalism/datalog/expression_properties.hpp"
#include "tyr/formalism/datalog/interning.hpp"
#include "tyr/formalism/datalog/repository.hpp"
#include "tyr/formalism/datalog/views.hpp"
#include "tyr/formalism/planning/merge_datalog.hpp"
#include "tyr/formalism/planning/repository.hpp"
#include "tyr/formalism/planning/views.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <yggdrasil/containers/unordered_set.hpp>
#include <utility>

namespace tyr::planning
{
namespace
{
using MetricFunctions = ygg::UnorderedSet<formalism::datalog::FunctionView<formalism::FluentTag>>;
using MetricEffects = ygg::DataList<formalism::datalog::NumericEffectOperator<LiftedTag, formalism::FluentTag>>;

template<formalism::FactKind T>
bool targets_metric_function(formalism::planning::NumericEffectOperatorView<LiftedTag, T> effect,
                             const MetricFunctions& functions, formalism::planning::MergeDatalogContext& context)
{
    return ygg::visit([&](auto arg) {
        return functions.contains(merge_p2d<T, formalism::FluentTag>(arg.get_fterm(), context).first.get_function());
    }, effect.get_variant());
}

bool is_real_conditional_effect(formalism::planning::ConditionalEffectView<LiftedTag> effect)
{
    const auto condition = effect.get_condition();
    return !effect.get_variables().empty() || !condition.get_literals<formalism::StaticTag>().empty()
        || !condition.get_literals<formalism::FluentTag>().empty() || !condition.get_literals<formalism::DerivedTag>().empty()
        || !condition.get_numeric_constraints().empty();
}

MetricEffects metric_effects(formalism::planning::ActionView<LiftedTag> action, const MetricFunctions& functions,
                             formalism::planning::MergeDatalogContext& context)
{
    MetricEffects result;
    for (const auto cond : action.get_effects()) {
        for (const auto effect : cond.get_effect().get_numeric_effects())
            if (targets_metric_function(effect, functions, context)) {
                if (is_real_conditional_effect(cond))
                    throw std::invalid_argument("GENERAL action costs with :conditional-effects are unsupported; compile conditional effects away first.");
                result.push_back(merge_p2d(effect, context));
            }
        if (const auto effect = cond.get_effect().get_auxiliary_numeric_effect())
            if (targets_metric_function(effect.value(), functions, context)) {
                if (is_real_conditional_effect(cond))
                    throw std::invalid_argument("GENERAL action costs with :conditional-effects are unsupported; compile conditional effects away first.");
                result.push_back(merge_p2d<formalism::AuxiliaryTag, formalism::FluentTag>(effect.value(), context));
            }
    }
    return result;
}

void append_variable(formalism::planning::VariableView variable,
                     formalism::planning::MergeDatalogContext& context,
                     ygg::Data<formalism::datalog::ConjunctiveCondition<LiftedTag>>& conj_cond)
{
    const auto index = merge_p2d(variable, context).first.get_index();
    if (std::ranges::find(conj_cond.variables, index) == conj_cond.variables.end())
        conj_cond.variables.push_back(index);
}

void fill_delete_free_condition(formalism::planning::ActionView<LiftedTag> action,
                                formalism::planning::ConditionalEffectView<LiftedTag> cond_eff,
                                TranslationContext<LiftedTag>& translation_context,
                                formalism::planning::MergeDatalogContext& context,
                                ygg::Data<formalism::datalog::ConjunctiveCondition<LiftedTag>>& conj_cond)
{
    // Action parameter may get deleted.
    for (const auto& variable : action.get_variables())
        append_variable(variable, context, conj_cond);
    for (const auto literal : action.get_condition().get_literals<formalism::StaticTag>())
        conj_cond.static_literals.push_back(merge_p2d(literal, translation_context.p2d.static_to_static_predicate, context).first.get_index());
    for (const auto literal : action.get_condition().get_literals<formalism::FluentTag>())
        if (literal.get_polarity())
            conj_cond.fluent_literals.push_back(merge_p2d(literal, translation_context.p2d.fluent_to_fluent_predicate, context).first.get_index());
    for (const auto numeric_constraint : action.get_condition().get_numeric_constraints())
        conj_cond.numeric_constraints.push_back(merge_p2d(numeric_constraint, context));

    for (const auto variable : cond_eff.get_variables())
        append_variable(variable, context, conj_cond);
    for (const auto literal : cond_eff.get_condition().template get_literals<formalism::StaticTag>())
        conj_cond.static_literals.push_back(merge_p2d(literal, translation_context.p2d.static_to_static_predicate, context).first.get_index());
    for (const auto literal : cond_eff.get_condition().template get_literals<formalism::FluentTag>())
        if (literal.get_polarity())
            conj_cond.fluent_literals.push_back(merge_p2d(literal, translation_context.p2d.fluent_to_fluent_predicate, context).first.get_index());
    for (const auto numeric_constraint : cond_eff.get_condition().get_numeric_constraints())
        conj_cond.numeric_constraints.push_back(merge_p2d(numeric_constraint, context));
}

auto create_delete_free_goal(formalism::planning::ConjunctiveConditionView<GroundTag> goal,
                             TranslationContext<LiftedTag>& translation_context,
                             formalism::planning::MergeDatalogContext& context)
{
    auto conj_cond = formalism::datalog::checkout<formalism::datalog::ConjunctiveCondition<GroundTag>>(context.builder());

    for (const auto fact : goal.get_facts<formalism::PositiveTag>())
        if (const auto literal = merge_p2d(fact, true, translation_context.p2d.fluent_to_fluent_predicate, context))
            conj_cond->fluent_literals.push_back(literal->get_index());

    for (const auto numeric_constraint : goal.get_numeric_constraints())
        conj_cond->numeric_constraints.push_back(merge_p2d(numeric_constraint, context));

    return formalism::datalog::insert(context.destination(), *conj_cond);
}

auto create_cond_effect_rule(formalism::planning::ActionView<LiftedTag> action,
                             formalism::planning::ConditionalEffectView<LiftedTag> cond_eff,
                             const MetricEffects& costs, bool general,
                             TranslationContext<LiftedTag>& translation_context,
                             formalism::planning::MergeDatalogContext& context)
{
    auto rule = formalism::datalog::checkout<formalism::datalog::Rule<LiftedTag, formalism::PredicateTag>>(context.builder());

    auto conj_cond = formalism::datalog::checkout<formalism::datalog::ConjunctiveCondition<LiftedTag>>(context.builder());

    fill_delete_free_condition(action, cond_eff, translation_context, context, *conj_cond);

    const auto new_conj_cond = formalism::datalog::insert(context.destination(), *conj_cond).first;

    ygg::extend(new_conj_cond.get_variables(), rule->variables);
    rule->body = new_conj_cond.get_index();
    for (const auto literal : cond_eff.get_effect().get_literals())
        if (literal.get_polarity())  /// ignore delete effects
            rule->heads.push_back(merge_p2d(literal.get_atom(), translation_context.p2d.fluent_to_fluent_predicate, context).first.get_index());
    rule->cost = general ? 0 : 1;
    rule->metric_effects.insert(rule->metric_effects.end(), costs.begin(), costs.end());

    return formalism::datalog::insert(context.destination(), *rule);
}

auto create_cond_numeric_effect_rule(formalism::planning::ActionView<LiftedTag> action,
                                     formalism::planning::ConditionalEffectView<LiftedTag> cond_eff,
                                     formalism::planning::NumericEffectOperatorView<LiftedTag, formalism::FluentTag> effect,
                                     const MetricEffects& costs, bool general,
                                     TranslationContext<LiftedTag>& translation_context,
                                     formalism::planning::MergeDatalogContext& context)
{
    auto rule = formalism::datalog::checkout<formalism::datalog::Rule<LiftedTag, formalism::FunctionTag>>(context.builder());

    auto conj_cond = formalism::datalog::checkout<formalism::datalog::ConjunctiveCondition<LiftedTag>>(context.builder());

    fill_delete_free_condition(action, cond_eff, translation_context, context, *conj_cond);

    const auto new_conj_cond = formalism::datalog::insert(context.destination(), *conj_cond).first;

    ygg::extend(new_conj_cond.get_variables(), rule->variables);
    rule->body = new_conj_cond.get_index();
    rule->head = merge_p2d(std::move(effect), context);
    rule->cost = general ? 0 : 1;
    rule->metric_effects.insert(rule->metric_effects.end(), costs.begin(), costs.end());

    return formalism::datalog::insert(context.destination(), *rule);
}

void translate_action_to_delete_free_rules(formalism::planning::ActionView<LiftedTag> action,
                                           ygg::Data<formalism::datalog::Program<LiftedTag>>& program,
                                           const MetricFunctions& functions, bool general,
                                           TranslationContext<LiftedTag>& translation_context,
                                           formalism::planning::MergeDatalogContext& context,
                                           RPGProgram<LiftedTag>::RuleToActionMappings& rule_to_action)
{
    const auto costs = general ? metric_effects(action, functions, context) : MetricEffects {};
    for (const auto cond_eff : action.get_effects())
    {
        if (std::ranges::any_of(cond_eff.get_effect().get_literals(), [](const auto literal) { return literal.get_polarity(); }))
        {
            const auto rule = create_cond_effect_rule(action, cond_eff, costs, general, translation_context, context).first;

            program.predicate_rules.push_back(rule.get_index());
            rule_to_action.predicate.emplace(rule.get_index(), action);
        }

        for (const auto numeric_effect : cond_eff.get_effect().get_numeric_effects())
        {
            const auto rule = create_cond_numeric_effect_rule(action, cond_eff, numeric_effect, costs, general, translation_context, context).first;

            program.function_rules.push_back(rule.get_index());
            rule_to_action.function.emplace(rule.get_index(), action);
        }
    }
}

auto create_program(formalism::planning::TaskView<LiftedTag> task, CostMode cost_mode,
                    TranslationContext<LiftedTag>& translation_context,
                    RPGProgram<LiftedTag>::RuleToActionMappings& rule_to_action,
                    formalism::datalog::Repository& destination)
{
    auto builder = formalism::datalog::Builder();
    auto context = formalism::planning::MergeDatalogContext(builder, destination);
    auto program = formalism::datalog::checkout<formalism::datalog::Program<LiftedTag>>(builder);

    for (const auto predicate : task.get_domain().get_predicates<formalism::StaticTag>())
    {
        const auto new_predicate = formalism::planning::merge_p2d(predicate, context).first;
        translation_context.d2p.static_to_static_predicate.emplace(new_predicate, predicate);
        translation_context.p2d.static_to_static_predicate.emplace(predicate, new_predicate);
        program->static_predicates.push_back(new_predicate.get_index());
    }
    for (const auto predicate : task.get_domain().get_predicates<formalism::FluentTag>())
    {
        const auto new_predicate = formalism::planning::merge_p2d(predicate, context).first;
        translation_context.d2p.fluent_to_fluent_predicate.emplace(new_predicate, predicate);
        translation_context.p2d.fluent_to_fluent_predicate.emplace(predicate, new_predicate);
        program->fluent_predicates.push_back(new_predicate.get_index());
    }

    for (const auto function : task.get_domain().get_functions<formalism::StaticTag>())
        program->static_functions.push_back(formalism::planning::merge_p2d(function, context).first.get_index());
    for (const auto function : task.get_domain().get_functions<formalism::FluentTag>())
    {
        const auto new_function = formalism::planning::merge_p2d(function, context).first;
        translation_context.p2d.fluent_to_fluent_function.emplace(function, new_function);
        program->fluent_functions.push_back(new_function.get_index());
    }

    if (cost_mode == CostMode::GENERAL)
        if (const auto function = task.get_domain().get_auxiliary_function())
        program->fluent_functions.push_back(merge_p2d<formalism::AuxiliaryTag, formalism::FluentTag>(function.value(), context).first.get_index());

    for (const auto object : task.get_domain().get_constants())
        program->objects.push_back(formalism::planning::merge_p2d(object, context).first.get_index());
    for (const auto object : task.get_objects())
        program->objects.push_back(formalism::planning::merge_p2d(object, context).first.get_index());

    for (const auto atom : task.get_atoms<formalism::StaticTag>())
        program->static_atoms.push_back(formalism::planning::merge_p2d(atom, translation_context.p2d.static_to_static_predicate, context).first.get_index());
    for (const auto atom : task.get_atoms<formalism::FluentTag>())
        program->fluent_atoms.push_back(formalism::planning::merge_p2d(atom, translation_context.p2d.fluent_to_fluent_predicate, context).first.get_index());

    for (const auto fterm_value : task.get_fterm_values<formalism::StaticTag>())
        program->static_fterm_values.push_back(formalism::planning::merge_p2d(fterm_value, context).first.get_index());
    for (const auto fterm_value : task.get_fterm_values<formalism::FluentTag>())
        program->fluent_fterm_values.push_back(formalism::planning::merge_p2d(fterm_value, context).first.get_index());

    if (cost_mode == CostMode::GENERAL)
        if (const auto value = task.get_auxiliary_fterm_value())
        program->fluent_fterm_values.push_back(merge_p2d<formalism::AuxiliaryTag, formalism::FluentTag>(value.value(), context).first.get_index());

    program->goal = create_delete_free_goal(task.get_goal(), translation_context, context).first.get_index();

    MetricFunctions functions;
    const bool general = cost_mode == CostMode::GENERAL;
    if (general && task.get_metric()) {
        const auto expr = merge_p2d(task.get_metric().value().get_fexpr(), context);
        auto terms = ygg::UnorderedSet<formalism::datalog::FunctionTermView<GroundTag, formalism::FluentTag>> {};
        formalism::datalog::collect_fterms(ygg::make_view(expr, context.destination()), terms);
        for (const auto term : terms)
            functions.insert(term.get_function());
    }
    for (const auto action : task.get_domain().get_actions())
        translate_action_to_delete_free_rules(action, *program, functions, general, translation_context, context, rule_to_action);

    return formalism::datalog::insert(destination, *program).first;
}

struct RPGTables
{
    TranslationContext<LiftedTag> translation_context;
    RPGProgram<LiftedTag>::RuleToActionMappings rule_to_action;

    RPGTables remap(SymbolRemapper& remapper) const
    {
        auto result = RPGTables { remapper.translation_context(translation_context), {} };
        for (const auto& [rule, action] : rule_to_action.predicate)
            result.rule_to_action.predicate.emplace(remapper.rule(rule), action);
        for (const auto& [rule, action] : rule_to_action.function)
            result.rule_to_action.function.emplace(remapper.rule(rule), action);
        return result;
    }
};

auto create_datalog_program(formalism::planning::TaskView<LiftedTag> task,
                            CostMode cost_mode,
                            bool split_rules,
                            TranslationContext<LiftedTag>& translation_context,
                            RPGProgram<LiftedTag>::RuleToActionMappings& rule_to_action,
                            datalog::RuleSplitting& rule_splitting,
                            RPGProgram<LiftedTag>::MetricTargets& metric_targets)
{
    auto factory = std::make_shared<formalism::datalog::RepositoryFactory>();
    const auto num_objects = task.get_num_objects();
    auto repository = factory->create_shared(num_objects);
    auto tables = RPGTables {};
    const auto program = create_program(task, cost_mode, tables.translation_context, tables.rule_to_action, *repository);
    auto result = TranslatedProgram<RPGTables> { program, std::move(repository), std::move(tables) };

    if (split_rules)
    {
        result = transform(result, *factory, num_objects, decompose_static_literals_pass());
        result = transform(result,
                           *factory,
                           num_objects,
                           [&](formalism::datalog::ProgramView<LiftedTag> source, formalism::datalog::Repository& destination, datalog::RuleMapping& rules)
                           {
                               const auto output = datalog::split_rules(source, destination, rule_splitting);
                               rules = rule_splitting.rules;
                               return output;
                           });
    }

    if (cost_mode == CostMode::GENERAL)
        if (const auto metric = task.get_metric()) {
            auto builder = formalism::datalog::Builder {};
            auto context = formalism::planning::MergeDatalogContext { builder, *result.repository };
            const auto expr = merge_p2d(metric.value().get_fexpr(), context);
            auto terms = ygg::UnorderedSet<formalism::datalog::FunctionTermView<GroundTag, formalism::FluentTag>> {};
            formalism::datalog::collect_fterms(ygg::make_view(expr, *result.repository), terms);
            for (const auto term : terms)
                metric_targets.insert(term.get_row().get_index());
        }

    translation_context = std::move(result.tables.translation_context);
    rule_to_action = std::move(result.tables.rule_to_action);
    return datalog::Program<LiftedTag>(result.program, std::move(result.repository), std::move(factory));
}

}

template<>
RPGProgram<LiftedTag>::RPGProgram(formalism::planning::TaskView<LiftedTag> task, CostMode cost_mode, bool split_rules) :
    m_translation_context(),
    m_rule_to_action(),
    m_rule_splitting(),
    m_datalog_program(create_datalog_program(task, cost_mode, split_rules, m_translation_context, m_rule_to_action, m_rule_splitting, m_metric_targets))
{
    // std::cout << m_datalog_program.get_program() << std::endl;
}

}
