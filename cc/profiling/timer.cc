#include "timer.h"

#include <cassert>
#include <chrono>

using namespace std::chrono;

long long int now() {
  auto time = system_clock::now().time_since_epoch();
  return duration_cast<nanoseconds>(time).count();
}

Timer::Timer() : timer_on_(false), t_start_nano_(0), elapsed_nano_(0) {}

void Timer::start() {
  assert(!timer_on_);
  t_start_nano_ = now();
  timer_on_ = true;
}

void Timer::stop() {
  assert(timer_on_);
  double t_end_nano = now();
  elapsed_nano_ += (t_end_nano - t_start_nano_);
  timer_on_ = false;
}

void Timer::clear() {
  assert(!timer_on_);
  elapsed_nano_ = 0;
}

double Timer::elapsed_sec() const {
  assert(!timer_on_);
  return elapsed_nano_ * 1e-9;
}