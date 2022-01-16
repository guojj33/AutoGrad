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
/* 一个GradVar与一个_Float对应，记录其梯度值，以及_Float是由哪个GradOp对应的正向操作计算出来的 */
public:
  float data = 0;
  GradOp * grad_op = nullptr;

  GradVar() {
    // cout << typeid(*this).name() << " " << this << " is being created.\n";
  }

  void delete_grad_op () {
    if (grad_op != nullptr) {
      delete grad_op;
      grad_op = nullptr;
    }
  }

  ~GradVar() {
    cout << typeid(*this).name() << " " << this << " is being deleted.\n";
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

  _Float(float d, bool stop_grad=false) {
    this->stop_grad = stop_grad;
    if (!stop_grad)
      grad = new GradVar();
    this->data = d;
    cout << typeid(*this).name() << " " << this << " is being created.";
    cout << " data=" << this->data << ".\n";
  }

  ~_Float() {
    cout << typeid(*this).name() << " " << this << " is being deleted.";
    cout << " data=" << this->data << ".\n";
    if (grad != nullptr) {
      delete grad;
      grad = nullptr;
    }
  }

  void printGradient() {
    if (!stop_grad) {
      cout << grad->data << "\n";
    }
    else {
      cout << "none\n";
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
};

queue<_Float *> global__Float_to_delete;

void global__Float_clear() {
  _Float *front = global__Float_to_delete.front();
  global__Float_to_delete.pop();
  front->grad->delete_grad_op(); // 释放方向传播其实点的GradOp，但不释放_Float本身和GradVar
  cout << "front of queue: " << front << "\n";
  
  // 释放中间变量_Float、GradVar、GradOp
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
      if (a->grad->grad_op != nullptr)
        global_grad_pending_ops.push(a->grad->grad_op);
    }
    if (!b->stop_grad) {
      update_b_grad();
      if (b->grad->grad_op != nullptr)
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
    cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& operator+(_Float &a, _Float &b) {
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
    cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& operator-(_Float &a, _Float &b) {
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
    cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& operator*(_Float &a, _Float &b) {
  _Float * c = new _Float(a.data*b.data);
  MulBackward * mul_op = new MulBackward(&a, &b, c);
  c->grad->grad_op = (GradOp*)mul_op;
  return *c;
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
      if (x->grad->grad_op != nullptr) {
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
    cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }
};

_Float& sigmoid(_Float &x) {
  float y_data = 1.0/(1+exp(-x.data));
  _Float * y = new _Float(y_data);
  SigmoidBackward * sigmoid_op = new SigmoidBackward(&x, y);
  y->grad->grad_op = (GradOp*)sigmoid_op;
  return *y;
}