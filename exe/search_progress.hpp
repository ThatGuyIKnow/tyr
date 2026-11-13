#ifndef TYR_EXE_SEARCH_PROGRESS_HPP_
#define TYR_EXE_SEARCH_PROGRESS_HPP_

#include <boost/json.hpp>
#include <bit>
#include <filesystem>
#include <fstream>
#include <map>
#include <unistd.h>

namespace tyr::cli {

class SearchProgress {
    using Clock = std::chrono::steady_clock;
    std::filesystem::path m_plan;
    std::ofstream m_layers, m_setup;
    Clock::time_point m_start = Clock::now(), m_stage = m_start;
    bool m_has_plan = false;
    bool m_grounded;
    boost::json::object m_stage_memory;

    static boost::json::object memory() {
        std::ifstream input("/proc/self/statm");
        std::uint64_t pages = 0, resident = 0;
        input >> pages >> resident;
        return {{"rss_bytes", resident * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE))},
                {"peak_rss_bytes", ygg::get_peak_memory_usage_in_bytes()}, {"scope", "shared_process"}};
    }
    static void append(std::ofstream& out, const boost::json::object& record) {
        out << boost::json::serialize(record) << '\n' << std::flush;
        if (!out) throw std::runtime_error("Could not write search progress log");
    }
    template<typename Task>
    static boost::json::object fdr(const Task& task) {
        std::map<std::size_t, std::uint64_t> histogram;
        std::uint64_t count = 0, values = 0, bits = 0;
        for (const auto variable : task.get_fdr_context()->get_variables()) {
            const auto size = variable.get_domain_size();
            ++count; values += size; bits += std::bit_width(size - 1); ++histogram[size];
        }
        boost::json::array domains;
        for (const auto& [size, variables] : histogram) domains.push_back(boost::json::object{{"domain_size", size}, {"variables", variables}});
        return {{"variables", count}, {"values", values}, {"packed_bits", bits}, {"domain_histogram", domains}};
    }
    static boost::json::object worker(const planning::Statistics& stats, std::size_t index) {
        return {{"worker", index}, {"expanded", stats.get_num_expanded()},
                {"pruned", stats.get_num_pruned()}, {"deadends", stats.get_num_deadends()},
                {"idle_time_ns", stats.get_idle_time().count()},
                {"search_time_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(stats.get_search_time()).count()},
                {"generated_states", stats.get_num_generated_successors()}, {"generated_candidates", stats.get_num_generated_candidates()},
                {"registered_states", stats.get_num_registered_states()}, {"state_storage_bytes", stats.get_state_storage_memory_usage()},
                {"transferred_candidates", stats.get_num_transferred_candidates()}, {"mpi_sent_states", 0}};
    }
public:
    explicit SearchProgress(const std::string& plan, bool grounded) : m_plan(plan), m_grounded(grounded), m_stage_memory(memory()) {
        auto directory = m_plan.parent_path();
        if (directory.empty()) directory = ".";
        std::filesystem::create_directories(directory);
        m_layers.open(directory / "f-completion.txt");
        m_setup.open(directory / "setup.jsonl");
        if (!m_layers || !m_setup) throw std::runtime_error("Could not open search progress logs");
        append(m_setup, {{"event", "process_start"}, {"total_time_ns", 0}, {"memory", memory()}});
    }
    bool has_plan() const noexcept { return m_has_plan; }
    void begin_stage(const char* name = nullptr) {
        m_stage = Clock::now(); m_stage_memory = memory();
        if (name) append(m_setup, {{"event", std::string(name) + "_started"}, {"stage_time_ns", 0},
            {"total_time_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(m_stage - m_start).count()}, {"memory", m_stage_memory}});
    }
    template<typename Task>
    void stage(const char* name, const Task& task) {
        const auto now = Clock::now();
        append(m_setup, {{"event", name}, {"stage_time_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_stage).count()},
                        {"total_time_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_start).count()},
                        {"representation", m_grounded ? "grounded" : "lifted"}, {"search_time_ns", 0}, {"expanded", 0},
                        {"generated_states", 0}, {"mpi_sent_states", 0}, {"memory_before", m_stage_memory}, {"memory", memory()}, {"fdr", fdr(task)}});
        begin_stage();
    }
    template<typename Kind>
    void attach(planning::astar_eager::Options<Kind>& options, const planning::TaskPtr<Kind>& task) {
        options.progress_callback = [this, task](std::optional<ygg::float_t> layer, const planning::Plan<Kind>* plan,
                                               const planning::Statistics& stats, const std::vector<planning::Statistics>& workers) {
            boost::json::array by_worker;
            for (std::size_t i = 0; i < workers.size(); ++i) by_worker.push_back(worker(workers[i], i));
            auto record = worker(stats, 0);
            record.erase("worker");
            record["event"] = plan ? "optimal_plan" : "f_layer_completed";
            record["representation"] = m_grounded ? "grounded" : "lifted";
            record["generated_states_definition"] = "accepted_successors_excluding_initial_state";
            record["search_time_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(stats.get_search_time()).count();
            record["total_time_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_start).count();
            record["memory"] = memory(); record["workers"] = std::move(by_worker); record["fdr"] = fdr(*task);
            if (layer) { record["f"] = *layer; append(m_layers, record); }
            if (plan) {
                record["cost"] = plan->get_cost(); record["plan_file"] = m_plan.string();
                auto temporary = m_plan; temporary += ".tmp";
                { std::ofstream out(temporary); out << fmt::format("{}", *plan); out.flush();
                  if (!out) throw std::runtime_error("Could not write proven-optimal plan"); }
                std::filesystem::rename(temporary, m_plan);
                std::ofstream out(m_plan.parent_path() / "optimal-plan.json");
                append(out, record); m_has_plan = true;
            }
        };
    }
};
}
#endif
