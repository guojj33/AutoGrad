#include <iostream>
using namespace std;

class base {
public:
  virtual void update() = 0; 

  void run() {
    cout << "start of run\n";
    update();
    cout << "end of run\n";
  }
};

class derive : public base {
public:
  void update() {
    cout << "derive update()\n";
  }
};

void f(base *x) {
  x->run();
}

int main () {
  derive *d = new derive();
  f(d);
  delete d;
}