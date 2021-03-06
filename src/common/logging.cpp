#include "logging.h"
#include "training/config.h"

std::shared_ptr<spdlog::logger> stderrLogger(const std::string& name,
                                             const std::string& pattern,
                                             const std::vector<std::string>& files) {
  std::vector<spdlog::sink_ptr> sinks;

  auto stderr_sink = spdlog::sinks::stderr_sink_mt::instance();
  sinks.push_back(stderr_sink);

  for(auto&& file : files) {
    auto file_sink = std::make_shared<spdlog::sinks::simple_file_sink_st>(file, true);
    sinks.push_back(file_sink);
  }

  auto logger = std::make_shared<spdlog::logger>(name, begin(sinks), end(sinks));

  spdlog::register_logger(logger);
  logger->set_pattern(pattern);
  return logger;
}

void createLoggers(const marian::Config& options) {
      
  std::vector<std::string> generalLogs;
  std::vector<std::string> validLogs;
  if(options.has("log")) {
    generalLogs.push_back(options.get<std::string>("log"));
    validLogs.push_back(options.get<std::string>("log"));
  }

  if(options.has("valid-log")) {
    validLogs.push_back(options.get<std::string>("valid-log"));
  }

  Logger info{stderrLogger("info", "[%Y-%m-%d %T] %v", generalLogs)};
  Logger config{stderrLogger("config", "[%Y-%m-%d %T] [config] %v", generalLogs)};
  Logger memory{stderrLogger("memory", "[%Y-%m-%d %T] [memory] %v", generalLogs)};
  Logger data{stderrLogger("data", "[%Y-%m-%d %T] [data] %v", generalLogs)};
  Logger valid{stderrLogger("valid", "[%Y-%m-%d %T] [valid] %v", validLogs)};
  Logger translate{stderrLogger("translate", "%v")};
}
