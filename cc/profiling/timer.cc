#include "timer.h"

#include <chrono>
#include <stdexcept>

using namespace std::chrono;

long long int now() {
  auto time = system_clock::now().time_since_epoch();
  return duration_cast<nanoseconds>(time).count();
}

Timer::Timer() : timer_on_(false), t_start_nano_(0), elapsed_nano_(0) {}

void Timer::start() {
  if (timer_on_) throw std::runtime_error("Timer is already running.");
  t_start_nano_ = now();
  timer_on_ = true;
}

void Timer::stop() {
  if (!timer_on_) throw std::runtime_error("Timer is already stopped.");
  double t_end_nano = now();
  elapsed_nano_ += (t_end_nano - t_start_nano_);
  timer_on_ = false;
}

void Timer::clear() {
  if (timer_on_) throw std::runtime_error("Cannot clear running timer.");
  elapsed_nano_ = 0;
}

double Timer::elapsed_sec() const {
  if (timer_on_) throw std::runtime_error("Timer is still running.");
  return elapsed_nano_ * 1e-9;
}