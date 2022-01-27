#ifndef FLOAT_H
#define FLOAT_H

#include "_Float.h"

class Float {
public:
  _Float *data = nullptr;

  Float() {
    data = new _Float(0, false);
    cout << typeid(*this).name() << " " << this << " is being created.\n";
  }

  Float(float d, bool stop_gradient=false) {
    data = new _Float(d, stop_gradient);
    // cout << typeid(*this).name() << " " << this << " is being created.\n";
  }

  Float(_Float *data) {
    this->data = data;
    // cout << typeid(*this).name() << " " << this << " is being created.\n";
  }

  ~Float() {
    cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }

  void setValue(float d) {
    if (data != nullptr) {
      data->data = d;
    }
  }

  void clear() {
    if (this->data != nullptr)
      delete data;
    data = nullptr;
  }

  void printGradient() {
    data->printGradient();
    cout << "\n";
  }

  void backward() {
    data->backward();
    // 此时this->data已被释放
    this->data = nullptr;
  }

  float value() {
    return this->data->data;
  }

  friend Float operator+(const Float &a, const Float &b);
  friend Float operator-(const Float &a, const Float &b);
  friend Float operator*(const Float &a, const Float &b);
  friend Float sigmoid(const Float &x);
  friend Float pow(const Float &x, float a);
};

Float operator+(const Float &a, const Float &b) {
  _Float *c_data = &((*a.data)+(*b.data));
  return Float(c_data);
}

Float operator-(const Float &a, const Float &b) {
  _Float *c_data = &((*a.data)-(*b.data));
  return Float(c_data);
}

Float operator*(const Float &a, const Float &b) {
  _Float *c_data = &((*a.data)*(*b.data));
  return Float(c_data);
}

Float sigmoid(const Float &x) {
  _Float *y_data = &(sigmoid(*x.data));
  return Float(y_data);
}

Float pow(const Float &x, float a) {
  _Float *y_data = &(pow(*x.data, a));
  return Float(y_data);
}

#endif