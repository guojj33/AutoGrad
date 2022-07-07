#ifndef _FLOAT_H
#define _FLOAT_H

#include <queue>
#include <cmath>
#include <iostream>

using namespace std;

class GradOp {
/* GradOp 在正向运算时被创建，在反向传播时run会被调用以计算梯度 */
public:
  virtual void run() = 0;
  virtual ~GradOp() {};
};

void global__Float_clear();

queue<GradOp*> global_grad_pending_ops;

void global_grad_cal() {
  while (!global_grad_pending_ops.empty()) {
    GradOp * cur_grad_op = global_grad_pending_ops.front();
    global_grad_pending_ops.pop();
    cur_grad_op->run();
  }
  global__Float_clear();
}

class GradVar {
/* 一个GradVar与一个_Float对应，记录其梯度值 */
public:
  float data = 0; // 梯度值
  GradOp * grad_op = nullptr; // GradVar对应的_Float是由此grad_op对应的正向运算计算出来的
  float depend_count = 0; // 输入中包含此_Float的正向“运算”的个数，_Float的完整梯度的计算依赖于这些“运算”

  GradVar() {
    // //cout << typeid(*this).name() << " " << this << " is being created.\n";
  }

  void increase_depend_count() {
    depend_count += 1;
  }

  void decrease_depend_count() {
    depend_count -= 1;
  }

  bool is_gradient_complete () {
    return depend_count == 0;
  }

  void delete_grad_op () {
    if (grad_op != nullptr) {
      delete grad_op;
      grad_op = nullptr;
    }
  }

  ~GradVar() {
    //cout << typeid(*this).name() << " " << this << " is being deleted.\n";
    if (grad_op != nullptr) {
      delete grad_op;
      grad_op = nullptr;
    }
  }
};

class _Float {
public:
  float data = 0;
  GradVar * grad = nullptr;
  bool stop_grad = false;

  _Float(bool stop_grad=false) {
    grad = new GradVar();
    this->stop_grad = stop_grad;
    //cout << typeid(*this).name() << " " << this << " is being created with _Float().\n";
  }

  _Float(float d, bool stop_grad=false) {
    this->stop_grad = stop_grad;
    if (!stop_grad)
      grad = new GradVar();
    this->data = d;
    //cout << typeid(*this).name() << " " << this << " is being created.";
    //cout << " data=" << this->data << ".\n";
  }

  void increase_grad_depend_count() {
    if (grad != nullptr) {
      grad->increase_depend_count();
    }
  }

  void decrease_grad_depend_count() {
    if (grad != nullptr) {
      grad->decrease_depend_count();
    }
  }

  bool is_grad_complete() {
    if (grad != nullptr) {
      return grad->is_gradient_complete();
    }
    return false;
  }

  ~_Float() {
    //cout << typeid(*this).name() << " " << this << " is being deleted.";
    //cout << " data=" << this->data << ".\n";
    if (grad != nullptr) {
      delete grad;
      grad = nullptr;
    }
  }

  void printGradient() {
    if (!stop_grad) {
      cout << grad->data;
    }
    else {
      cout << "none";
    }
  }

  void backward() {
    if (!stop_grad) {
      grad->data = 1.0; // 自己对自己求导为1
      if (grad->grad_op != nullptr) {
        global_grad_pending_ops.push(grad->grad_op);
        global_grad_cal();
      }
    }
  }

  friend _Float& operator+(_Float &a, _Float &b);
  friend _Float& operator-(_Float &a, _Float &b);
  friend _Float& operator*(_Float &a, _Float &b);
  friend _Float& sigmoid(_Float &x);
  friend _Float& pow(_Float &x, float a);
};

queue<_Float *> global__Float_to_delete;

void global__Float_clear() {
  // 正向计算图中入度大于等于1的都被释放
  while(!global__Float_to_delete.empty()) {
    _Float * f = global__Float_to_delete.front();
    global__Float_to_delete.pop();
    delete f;
  }
}

class BinaryOp : GradOp {
public:
  // input
  _Float *a, *b;
  GradVar *c_grad;
  // output
  // GradVar *a_grad, *b_grad;

  _Float *c;

  BinaryOp(_Float *a, _Float *b, _Float *c) {
    this->a = a;
    this->b = b;
    this->c = c;
    this->c_grad = c->grad;
  }

  virtual void update_a_grad() = 0;
  virtual void update_b_grad() = 0;

  void run() override {
    if (!a->stop_grad) {
      update_a_grad();
      a->decrease_grad_depend_count();
      if (a->grad->grad_op != nullptr && a->is_grad_complete())
        global_grad_pending_ops.push(a->grad->grad_op);
    }
    if (!b->stop_grad) {
      update_b_grad();
      b->decrease_grad_depend_count();
      if (b->grad->grad_op != nullptr && b->is_grad_complete())
        global_grad_pending_ops.push(b->grad->grad_op);
    }
    global__Float_to_delete.push(c);
  }
};

class AddBackward : BinaryOp {
/* c = a + b */
public:
  AddBackward(_Float *a, _Float *b, _Float *c) : BinaryOp(a, b, c) {}

  void update_a_grad() override {
    a->grad->data += c_grad->data*1;
  }

  void update_b_grad() override {
    b->grad->data += c_grad->data*1;
  }

  ~AddBackward() {
    //cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& operator+(_Float &a, _Float &b) {
  a.increase_grad_depend_count();
  b.increase_grad_depend_count();
  _Float * c = new _Float(a.data+b.data);
  AddBackward * add_op = new AddBackward(&a, &b, c);
  c->grad->grad_op = (GradOp*)add_op;
  return *c;
}

class SubBackward : BinaryOp {
/* c = a - b */
public:
  SubBackward(_Float *a, _Float *b, _Float *c) : BinaryOp(a, b, c) {}

  void update_a_grad() override {
    a->grad->data += c_grad->data*1;
  }

  void update_b_grad() override {
    b->grad->data += c_grad->data*(-1);
  }
  
  ~SubBackward() {
    //cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& operator-(_Float &a, _Float &b) {
  a.increase_grad_depend_count();
  b.increase_grad_depend_count();
  _Float * c = new _Float(a.data-b.data);
  SubBackward * sub_op = new SubBackward(&a, &b, c);
  c->grad->grad_op = (GradOp*)sub_op;
  return *c;
}

class MulBackward : BinaryOp {
/* c = a * b */
public:
  MulBackward(_Float *a, _Float *b, _Float *c) : BinaryOp(a, b, c) {}

  void update_a_grad() override {
    a->grad->data += c_grad->data * b->data;
  }

  void update_b_grad() override {
    b->grad->data += c_grad->data * a->data;
  }

  ~MulBackward() {
    //cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& operator*(_Float &a, _Float &b) {
  a.increase_grad_depend_count();
  b.increase_grad_depend_count();
  _Float * c = new _Float(a.data*b.data);
  MulBackward * mul_op = new MulBackward(&a, &b, c);
  c->grad->grad_op = (GradOp*)mul_op;
  return *c;
}

class DivBackward : BinaryOp {
/* c = a / b  */
public:
  DivBackward(_Float *a, _Float *b, _Float *c) : BinaryOp(a, b, c) {}

  void update_a_grad() override {
    a->grad->data += c_grad->data * (1.0 / b->data);
  }

  void update_b_grad() override {
    b->grad->data += c_grad->data * (-a->data/(b->data*b->data));
  }

  ~DivBackward() {

  }
};

_Float& operator/(_Float &a, _Float &b) {
  try {
    a.increase_grad_depend_count();
    b.increase_grad_depend_count();
    if (b.data == 0) {
      string info = "_Float division error: divided by zero\n";
      throw info;
    }
    _Float *c = new _Float(a.data/b.data);
    DivBackward * div_op = new DivBackward(&a, &b, c);
    c->grad->grad_op = (GradOp*)div_op;
    return *c;
  }
  catch(string info) {
    cout << info << "\n";
  }
}

class UnaryOp : GradOp {
public:
  // input
  _Float *x;
  GradVar *y_grad;
  // output
  // GradVar *x_grad;

  _Float *y;

  UnaryOp(_Float *x, _Float *y) {
    this->x = x;
    this->y = y;
    this->y_grad = y->grad;
  }

  virtual void update_x_grad() = 0;

  void run() override {
    if (!x->stop_grad) {
      update_x_grad();
      x->decrease_grad_depend_count();
      if (x->grad->grad_op != nullptr && x->is_grad_complete()) {
        global_grad_pending_ops.push(x->grad->grad_op);
      }
    }
    global__Float_to_delete.push(y);
  }
};

class SigmoidBackward : UnaryOp {
/* y = sigmoid(x) */
public:
  SigmoidBackward(_Float *x, _Float *y) : UnaryOp(x, y) {}

  void update_x_grad() override {
    x->grad->data += y_grad->data*(y->data)*(1-y->data);
  }

  ~SigmoidBackward() {
    //cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& sigmoid(_Float &x) {
  x.increase_grad_depend_count();
  float y_data = 1.0/(1+exp(-x.data));
  _Float * y = new _Float(y_data);
  SigmoidBackward * sigmoid_op = new SigmoidBackward(&x, y);
  y->grad->grad_op = (GradOp*)sigmoid_op;
  return *y;
}

class PowBackward: UnaryOp {
/* y = x^a */
public:
  float a;

  PowBackward(_Float *x, _Float *y, float a) : UnaryOp(x, y) {
    this->a = a;
  }

  void update_x_grad() override {
    x->grad->data += a*pow(x->data, a-1);
  }

  ~PowBackward() {
    //cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& pow(_Float &x, float a) {
  x.increase_grad_depend_count();
  float y_data = pow(x.data, a);
  _Float * y = new _Float(y_data);
  PowBackward * pow_op = new PowBackward(&x, y, a);
  y->grad->grad_op = (GradOp*)pow_op;
  return *y;
}

#endif