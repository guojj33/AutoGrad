#include <queue>
#include <cmath>
#include <iostream>

using namespace std;

class GradOp {
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

  friend _Float& operator*(_Float &a, _Float &b);
  friend _Float& operator+(_Float &a, _Float &b);
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
};

class MulBackward : BinaryOp {
public:
  MulBackward(_Float *a, _Float *b, _Float *c) : BinaryOp(a, b, c) {}

  void run() {
    if (!a->stop_grad) {
      a->grad->data += c_grad->data * b->data;
      if (a->grad->grad_op != nullptr)
        global_grad_pending_ops.push(a->grad->grad_op);
    }
    if (!b->stop_grad) {
      b->grad->data += c_grad->data * a->data;
      if (b->grad->grad_op != nullptr)
        global_grad_pending_ops.push(b->grad->grad_op);
    }
    global__Float_to_delete.push(c);
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

class AddBackward : BinaryOp {
public:
  AddBackward(_Float *a, _Float *b, _Float *c) : BinaryOp(a, b, c) {}

  void run() {
    if (!a->stop_grad) {
      a->grad->data += c_grad->data*1;
      if (a->grad->grad_op != nullptr)
        global_grad_pending_ops.push(a->grad->grad_op);
    }
    if (!b->stop_grad) {
      b->grad->data += c_grad->data*1;
      if (b->grad->grad_op != nullptr)
        global_grad_pending_ops.push(b->grad->grad_op);
    }
    global__Float_to_delete.push(c);
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

class SigmoidBackward : GradOp {
public:
  _Float *x;
  _Float *y;

  SigmoidBackward(_Float *x, _Float *y) {
    this->x = x;
    this->y = y;
  }

  void run() {
    if (!x->stop_grad) {
      x->grad->data += y->grad->data*(y->data)*(1-y->data);
      if (x->grad->grad_op != nullptr) {
        global_grad_pending_ops.push(x->grad->grad_op);
      }
    }
    global__Float_to_delete.push(y);
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