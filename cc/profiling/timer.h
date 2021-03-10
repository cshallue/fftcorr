#ifndef TIMER_H
#define TIMER_H

class Timer {
 public:
  Timer();
  void start();
  void stop();
  double elapsed_sec() const;
  void clear();

 private:
  bool timer_on_;
  long long int t_start_nano_;
  long long int elapsed_nano_;
};

#endif  // TIMER_H