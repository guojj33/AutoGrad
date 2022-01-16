#include "_Float.h"

class Float {
public:
  _Float *data = nullptr;

  Float(float d, bool stop_gradient=false) {
    data = new _Float(d, stop_gradient);
    // cout << typeid(*this).name() << " " << this << " is being created.\n";
  }

  Float(_Float *data) {
    this->data = data;
    // cout << typeid(*this).name() << " " << this << " is being created.\n";
  }

  ~Float() {
    // cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }

  void clear() {
    if (this->data != nullptr)
      delete data;
    data = nullptr;
  }

  void printGradient() {
    data->printGradient();
  }

  void backward() {
    data->backward();
  }

  friend Float operator*(const Float &a, const Float &b);
  friend Float operator+(const Float &a, const Float &b);
  friend Float sigmoid(const Float &x);
};

Float operator*(const Float &a, const Float &b) {
  _Float *c_data = &((*a.data)*(*b.data));
  return Float(c_data);
}

Float operator+(const Float &a, const Float &b) {
  _Float *c_data = &((*a.data)+(*b.data));
  return Float(c_data);
}

Float sigmoid(const Float &x) {
  _Float *y_data = &(sigmoid(*x.data));
  return Float(y_data);
}