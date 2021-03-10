#include "profiler.h"

void Profiler::start(const std::string& name) {
  // fprintf(stderr, "Starting %s\n", name.c_str());
  timers_[name].start();
}

void Profiler::stop(const std::string& name) {
  // fprintf(stderr, "Stopping %s\n", name.c_str());
  timers_.at(name).stop();  // throws if key doesn't exist
}

double Profiler::elapsed(const std::string& name) {
  // fprintf(stderr, "Elapsed %s\n", name.c_str());
  if (timers_.find(name) == timers_.end()) {
    return 0.0;
  }
  return timers_[name].elapsed_sec();
}