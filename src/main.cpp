#include "Float.h"

float sigmoid(float x) {
  return 1.0/(1+exp(-x));
}

int main () {
  Float a(1.0), b(2.0), c(3.0); // Float含有指向_Float的指针。Float构造时会new一个_Float，当Float销毁时，_Float仍会保留。
  cout << "start of cycle\n";
  for (int i = 0; i < 1; ++i) {
    Float d = pow((a*b-c),2);
    cout << "forward = " << d.value() << "\n";
    cout << "expect = " << pow(1*2-3, 2) << "\n";

    cout << "start of backward\n";
    d.backward();
    /* 启动反向传播。正向计算图中入度大于等于1的都被释放（就是除了输入样本和网络参数外都被释放）。*/
    cout << "end of backward\n";
    
    cout << "a_grad=";
    a.printGradient();
    cout << "b_grad=";
    b.printGradient();
    cout << "c_grad=";
    c.printGradient();
  }
  cout << "end of cycle\n";
  a.clear(); // 手动释放资源
  b.clear(); // 手动释放资源
  c.clear(); // 手动释放资源
  /* 为什么不让_Float随Float销毁而释放？ 
  如果这样实现，那么（a*b）产生的中间变量在语句结束后会被销毁？这样下一步执行backward时就会出错？ */
}