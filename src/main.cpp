#include "Float.h"
#include "Mat.h"

void test1 () {
  // Float含有指向_Float的指针。Float构造时会new一个_Float，当Float销毁时，_Float仍会保留。
  Float a(0.0);
  cout << "start of cycle\n";
  for (int i = 0; i < 1; ++i) {
    Float b = sigmoid(a);
    Float d = (b*b)+b;
    cout << d.value() << "\n";
    cout << "start of backward\n";
    d.backward();
    /* 启动反向传播。正向计算图中入度大于等于1的都被释放
    （就是除了输入样本和网络参数外的中间变量都被释放）。
    此时b和d的包含一个野指针，不要再使用。（怎么解决更好一点？）*/
    cout << "end of backward\n";
    
    cout << "a_grad=";
    a.printGradient();
  }
  cout << "end of cycle\n";
  // 手动释放资源
  a.clear(); 
  /* 为什么不让_Float随Float销毁而释放？ 
  如果那样实现，那么（a*b）产生的中间变量在语句结束后会被销毁？这样下一步执行backward时就会出错？ */
}

void test2() {
  Mat m1(1,2);
  m1.set(0, 0, 1);
  m1.set(0, 1, 2);
  m1.printMat();

  Mat m2(1,2);
  m2.set(0, 0, 2);
  m2.set(0, 1, 3);
  m2.printMat();

  Mat m3 = m1-m2+m1;
  m3.printMat();
  cout << "start of backward\n";
  m3.backward();
  cout << "end of backward\n";
  
  m1.printGradient(); // 2  2
  m2.printGradient(); // -1 -1

  m1.clear();
  m2.clear();
}

int main () {
  test1();
  cout << "\n";
  test2();
}