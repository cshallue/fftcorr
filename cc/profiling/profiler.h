#ifndef PROFILER_H
#define PROFILER_H

#include <map>
#include <string>

#include "timer.h"

class Profiler {
 public:
  void start(const std::string& name);
  void stop(const std::string& name);
  double elapsed(const std::string& name);

 private:
  std::map<std::string, Timer> timers_;
};

// Global Profiler object holding all counters.
extern Profiler profiler;

#endif  // PROFILER_H