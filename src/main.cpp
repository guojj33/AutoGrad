#include "Float.h"

int main () {
  Float a(1.0), b(2.0), c(3.0);
  cout << "start of cycle\n";
  for (int i = 0; i < 1; ++i) {
    Float d = sigmoid(a*b+c);
    cout << "start of backward\n";
    d.backward(); // 计算图中中间变量（a*b和（a*b+c））的_Float和GradVar将被释放，d的GradVar的GradOp被释放
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