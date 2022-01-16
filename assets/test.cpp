#include <iostream>
#include <exception>
using namespace std;

class t {
public:
  int data = 0;
  t *parent = nullptr;
  int link_count = 0; // 统计t当前被多少个t_cont指向，当为0时可delete本t

  void addLinkCount() {
    link_count += 1;
  }

  void reduceLinkCount() {
    link_count -= 1;
  }

  bool hasNoLinks() {
    return link_count == 0;
  }

  t() {
    // cout << typeid(*this).name() << " " << this << " is being created with t().\n";
  }

  t(int data) {
    this->data = data;
    // cout << typeid(*this).name() << " " << this << " data=" << this->data << ", is being created with t(int)\n";
  }

  ~t() {
    // cout << typeid(*this).name() << " " << this << " data=" << this->data << ", is being deleted with ~t()\n";
  }

  friend t& operator+(t &t1, t &t2);
};

t& operator+(t &t1, t &t2) {
  t *t3 = new t(t1.data+t2.data); // 堆
  t1.parent = t3;
  t2.parent = t3;
  return *t3;
}

// t_container，用局部对象t_cont来管理资源t，t_cont销毁时删除其所绑定的资源t
class t_cont {
public:
  t *tt;

  t_cont(t *tt) {
    this->tt = tt;
    tt->addLinkCount();
    cout << typeid(*this).name() << " " << this << ", data=" << tt->data << " is being created with t_cont(t).\n";
  }

  // copy constructor
  t_cont(const t_cont & _t_cont) {
    tt = _t_cont.tt;
    tt->addLinkCount();
    cout << typeid(*this).name() << " " << this << ", data=" << tt->data << " is being created with t_cont(const t_cont &).\n";
  }

  t_cont(int data) {
    tt = new t(data);
    tt->addLinkCount();
    cout << typeid(*this).name() << " " << this << ", data=" << tt->data << " is being created with t_cont(int).\n";
  }

  ~t_cont() {
    cout << typeid(*this).name() << " " << this << ", data=" << tt->data << " is being deleted.\n";
    if (tt != nullptr) {
      tt->reduceLinkCount();
      if (tt->hasNoLinks()) {
        delete tt;
      }
      tt = nullptr;
    }
  }

  friend t_cont operator+(const t_cont& t_cont_1, const t_cont& t_cont_2);

  t_cont& operator=(const t_cont& _t_cont) {
    cout << "start operator =\n";
    tt = _t_cont.tt;
    tt->addLinkCount();
    cout << "end of operator =\n";
    return *this;
  }
};

t_cont operator+(const t_cont& t_cont_1, const t_cont& t_cont_2) {
  cout << "start operator +\n";
  t *t3 = &((*(t_cont_1.tt))+(*(t_cont_2.tt)));
  t_cont t_cont_3(t3);
  cout << "end of operator +\n";
  return t_cont_3;
}

int main () {
  t_cont t1(1), t2(2), t3(4);
  for (int i = 0; i < 1; ++i) {
    t_cont t4 = (t1+t2)+t3;
    cout << "here\n";
  }
  cout << "end of cycle\n";
}