/*
 * Copyright (C) 2026 Dominik Drexler
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */
#ifndef TYR_SRC_PLANNING_ALGORITHMS_SEARCH_ENGINE_K_PARALLEL_HPP_
#define TYR_SRC_PLANNING_ALGORITHMS_SEARCH_ENGINE_K_PARALLEL_HPP_

#include "parallel.hpp"

namespace tyr::planning::detail
{

template<TaskKind Kind, SearchPolicyConcept<Kind> SearchPolicy>
class KParallelExecutionPolicy
{
    using Metadata = SuccessorMetadataT<SearchPolicy>;
    using BuilderPtr = ygg::SharedObjectPoolPtr<ygg::Builder<State<Kind>>, true>;

public:
    using SearchTag = ParallelSearch;
    static constexpr bool uses_owner_repositories = false;
    static constexpr bool uses_shared_search = true;
    struct WorkerState {};

    explicit KParallelExecutionPolicy(uint64_t) noexcept {}

    bool goal_is_proven(ygg::float_t cost) const noexcept { return m_synchronous && cost <= m_layer; }
    template<typename Engine> void wake_for_progress(Engine&) { m_condition.notify_all(); }
    template<typename Engine> ygg::float_t progress_priority(Engine&) const noexcept
    { return m_synchronous ? m_layer : std::numeric_limits<ygg::float_t>::infinity(); }

    static void validate(const OptionsT<SearchPolicy>& options)
    {
        if (options.num_search_workers == 0 || options.num_search_workers > std::numeric_limits<ygg::uint_t>::max())
            throw std::invalid_argument("K-parallel search requires a valid positive worker count.");
    }
    static size_t num_workers(const OptionsT<SearchPolicy>& options) noexcept { return options.num_search_workers; }
    static size_t search_node_divisor(const OptionsT<SearchPolicy>&) noexcept { return 1; }
    static ygg::Index<State<Kind>> search_node_index(ygg::Index<State<Kind>> state, ygg::Index<Worker>, size_t) noexcept { return state; }

    template<typename Engine>
    auto prepare_start_state(Engine& engine, const StateView<Kind>& state)
    {
        auto& root = engine.get_worker(ygg::Index<Worker>(0));
        return std::pair(ygg::Index<Worker>(0), materialize_state(state, *root.state_repository, *root.axiom_evaluator));
    }

    void initialize_best_h(ygg::float_t value) noexcept { m_best_h = value; }
    template<typename Callback>
    bool improve_best_h(ygg::float_t value, Callback&& callback)
    {
        const auto lock = std::lock_guard(m_best_h_mutex);
        if (value >= m_best_h)
            return false;
        m_best_h = value;
        std::forward<Callback>(callback)();
        return true;
    }

    void start(std::optional<std::chrono::steady_clock::time_point> deadline, ygg::float_t priority, const OptionsT<SearchPolicy>& options)
    {
        m_deadline = deadline;
        m_layer = priority;
        m_synchronous = SearchPolicy::synchronize_priority_layers(options);
        m_collect_destination_lock_statistics = options.collect_destination_lock_statistics;
        m_num_states.store(1, std::memory_order_relaxed);
    }

    template<typename Engine>
    void invoke(Engine& engine)
    {
        auto threads = std::vector<std::jthread> {};
        try
        {
            threads.reserve(engine.num_workers());
            for (size_t i = 0; i < engine.num_workers(); ++i)
                threads.emplace_back([this, &engine, i] {
                    try { engine.worker_loop(engine.get_worker(ygg::Index<Worker>(static_cast<ygg::uint_t>(i)))); }
                    catch (...) { record_exception(std::current_exception()); }
                });
        }
        catch (...) { record_exception(std::current_exception()); }
    }

    template<typename Engine, typename WorkerData>
    bool begin_iteration(Engine&, WorkerData&) const noexcept { return running(); }
    bool running() const noexcept { return m_status.load(std::memory_order_acquire) == SearchStatus::IN_PROGRESS; }
    bool timed_out() const noexcept { return m_deadline && std::chrono::steady_clock::now() >= *m_deadline; }
    SearchStatus status() const noexcept { return m_status.load(std::memory_order_acquire); }
    ygg::float_t incumbent_cost() const noexcept { return m_incumbent.load(std::memory_order_relaxed); }
    std::optional<WorkerStateIndex<Kind>> goal() const { const auto lock = std::lock_guard(m_terminal_mutex); return m_goal; }
    std::exception_ptr exception() const { const auto lock = std::lock_guard(m_terminal_mutex); return m_exception; }
    void set_terminal(SearchStatus status)
    {
        auto expected = SearchStatus::IN_PROGRESS;
        m_status.compare_exchange_strong(expected, status, std::memory_order_acq_rel);
        m_condition.notify_all();
    }
    bool consider_goal(WorkerStateIndex<Kind> goal, ygg::float_t cost, bool terminate)
    {
        const auto lock = std::lock_guard(m_terminal_mutex);
        if (!running() || cost >= m_incumbent.load(std::memory_order_relaxed))
            return false;
        m_goal = goal;
        m_incumbent.store(cost, std::memory_order_relaxed);
        if (terminate)
            set_terminal(SearchStatus::SOLVED);
        return true;
    }

    template<typename Engine, typename WorkerData>
    bool can_expand_locked(Engine& engine, WorkerData& worker)
    {
        auto& search = worker.search_policy();
        if (!running() || search.empty() || engine.progress_requested())
            return false;
        const auto priority = search.get_min_priority();
        if (priority > incumbent_cost())
            return false;
        if (m_synchronous && priority > m_layer)
        {
            if (m_inflight != 0)
                return false;
            engine.on_finish_priority_layer(m_layer);
            m_layer = priority;
        }
        ++m_inflight;
        return true;
    }

    template<typename WorkerData, typename Callback>
    decltype(auto) with_worker_lock(WorkerData&, Callback&& callback)
    {
        auto lock = std::unique_lock(m_mutex);
        // Keep the claimed node stable until it has been marked closed.
        const auto evaluate_locked = [&]<typename Evaluate>(Evaluate&& evaluate) { return std::forward<Evaluate>(evaluate)(); };
        return std::forward<Callback>(callback)(evaluate_locked);
    }

    template<typename Engine, typename WorkerData>
    AcceptanceResult route(Engine& engine, WorkerData& sender, const Node<Kind>& source, BuilderPtr target, const StateDelta& delta,
                           ygg::float_t auxiliary_value, formalism::planning::ActionBindingView action, Metadata metadata)
    {
        const auto metric = engine.evaluate_successor_metric(*target, auxiliary_value);
        const auto g = compute_successor_g_value(metadata.source_g_value, metric, engine.m_options->cost_mode);
        if (!std::isfinite(g))
            throw std::runtime_error("find_solution(...): successor path cost is not finite.");
        auto state = sender.state_repository->register_state(*sender.axiom_evaluator, std::move(target), source.get_state(), delta);
        auto routed = RoutedSuccessorT<Engine> { LabeledNode<Kind> { action, Node<Kind>(state, metric) }, std::move(metadata), g, false };
        sender.statistics.increment_num_generated_candidates(false);
        auto wait_start = m_collect_destination_lock_statistics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
        auto lock = std::unique_lock(m_mutex);
        auto hold_start = m_collect_destination_lock_statistics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
        const auto record_lock = [&] {
            if (m_collect_destination_lock_statistics)
                sender.statistics.add_destination_lock_statistics(std::chrono::duration_cast<std::chrono::nanoseconds>(hold_start - wait_start),
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - hold_start));
        };
        const auto evaluate_heuristic = [&](const StateView<Kind>& candidate) {
            record_lock();
            lock.unlock();
            const auto h = sender.heuristic->evaluate(candidate);
            wait_start = m_collect_destination_lock_statistics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
            lock.lock();
            hold_start = m_collect_destination_lock_statistics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
            return h;
        };
        const auto result = running() ? engine.accept_successor(sender, source, routed, evaluate_heuristic) : AcceptanceResult::TERMINAL;
        record_lock();
        lock.unlock();
        m_condition.notify_all();
        return result;
    }

    template<typename Engine, typename WorkerData, typename EmitTransition>
    std::optional<AcceptanceResult> accept_generated_goal(Engine&, WorkerData&, SearchNodeT<SearchPolicy>&,
                                                          const RoutedSuccessorT<Engine>&, const StateView<Kind>&,
                                                          ygg::float_t, EmitTransition&&) { return std::nullopt; }
    template<typename Engine, typename WorkerData>
    static bool is_generated_goal(Engine&, WorkerData&, const RoutedSuccessorT<Engine>&, const StateView<Kind>&) noexcept { return false; }

    bool reserve_state(ygg::uint_t limit) noexcept
    {
        auto count = m_num_states.load(std::memory_order_relaxed);
        while (count < limit)
            if (m_num_states.compare_exchange_weak(count, count + 1, std::memory_order_relaxed))
                return true;
        return false;
    }

    template<typename Engine>
    void finish_expansion(Engine&)
    {
        const auto lock = std::lock_guard(m_mutex);
        assert(m_inflight > 0);
        --m_inflight;
        m_condition.notify_all();
    }

    template<typename WorkerData>
    void finish_worker(WorkerData& worker)
    {
        const auto lock = std::lock_guard(m_mutex);
        worker.statistics.set_search_end_time_point(std::chrono::steady_clock::now());
    }

    template<typename Engine, typename WorkerData>
    void wait_for_work(Engine& engine, WorkerData& worker)
    {
        const auto idle_start = std::chrono::steady_clock::now();
        auto lock = std::unique_lock(m_mutex);
        const auto record_idle = [&] {
            worker.statistics.add_idle_time(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - idle_start));
        };
        while (running())
        {
            if (engine.progress_requested()) { record_idle(); return; }
            if (timed_out())
            {
                set_terminal(SearchStatus::OUT_OF_TIME);
                record_idle();
                return;
            }
            const auto& search = worker.search_policy();
            const auto priority = search.get_min_priority();
            if (m_inflight == 0 && (search.empty() || priority > incumbent_cost()))
            {
                set_terminal(std::isfinite(incumbent_cost()) ? SearchStatus::SOLVED : SearchStatus::EXHAUSTED);
                record_idle();
                return;
            }
            if (!search.empty() && priority <= incumbent_cost() && (!m_synchronous || priority <= m_layer || m_inflight == 0))
            {
                record_idle();
                return;
            }
            if (m_deadline)
            {
                if (m_condition.wait_until(lock, *m_deadline) == std::cv_status::timeout && running())
                    set_terminal(SearchStatus::OUT_OF_TIME);
            }
            else
                m_condition.wait(lock);
        }
        record_idle();
    }

    template<typename Engine> void notify_if_stopped(Engine&)
    {
        if (!running())
        {
            const auto lock = std::lock_guard(m_mutex);
            m_condition.notify_all();
        }
    }
    template<typename Engine>
    static void snapshot_worker_state_statistics(Engine& engine)
    {
        auto& root = engine.get_worker(ygg::Index<Worker>(0));
        detail::snapshot_state_repository_statistics(*root.state_repository, root.statistics);
    }

private:
    void record_exception(std::exception_ptr exception)
    {
        {
            const auto lock = std::lock_guard(m_terminal_mutex);
            if (!m_exception)
                m_exception = std::move(exception);
            m_status.store(SearchStatus::FAILED, std::memory_order_release);
        }
        const auto lock = std::lock_guard(m_mutex);
        m_condition.notify_all();
    }

    std::mutex m_mutex;
    std::condition_variable m_condition;
    mutable std::mutex m_terminal_mutex;
    std::mutex m_best_h_mutex;
    std::atomic<SearchStatus> m_status { SearchStatus::IN_PROGRESS };
    std::atomic<ygg::float_t> m_incumbent { std::numeric_limits<ygg::float_t>::infinity() };
    std::atomic<ygg::uint_t> m_num_states { 0 };
    std::optional<std::chrono::steady_clock::time_point> m_deadline;
    std::optional<WorkerStateIndex<Kind>> m_goal;
    std::exception_ptr m_exception;
    ygg::float_t m_best_h { std::numeric_limits<ygg::float_t>::infinity() };
    ygg::float_t m_layer { std::numeric_limits<ygg::float_t>::infinity() };
    size_t m_inflight = 0;
    bool m_synchronous = false;
    bool m_collect_destination_lock_statistics = false;
};

}
#endif
