#include "Float.h"

float sigmoid(float x) {
  return 1.0/(1+exp(-x));
}

int main () {
  Float a(1.0), b(2.0), c(3.0); // Float含有指向_Float的指针。Float构造时会new一个_Float，当Float销毁时，_Float仍会保留。
  cout << "start of cycle\n";
  for (int i = 0; i < 1; ++i) {
    Float d = sigmoid(a*b-c);
    cout << "forward = " << d.value() << "\n";
    cout << "expect = " << sigmoid(1*2-3) << "\n";

    cout << "start of backward\n";
    d.backward(); 
    /* 启动反向传播。计算图中的中间变量（a*b和（a*b+c））的_Float及其GradVar及其GradOp都将被释放，
    而d只有其_Float的GradVar的GradOp被释放。*/
    cout << "end of backward\n";
    
    a.printGradient();
    b.printGradient();
    c.printGradient();
    d.printGradient();

    d.clear();
    /* 输入输出不会自动释放，要调用clear手动释放。只有Float计算产生的中间变量在backward时会自动释放。 */
    /* 为什么不让_Float随Float销毁而释放？ 
    如果这样实现，那么（a*b）产生的中间变量在语句结束后会被销毁？这样下一步执行backward时就会出错？ */
  }
  cout << "end of cycle\n";
  a.clear(); // 手动释放资源
  b.clear(); // 手动释放资源
  c.clear(); // 手动释放资源
}