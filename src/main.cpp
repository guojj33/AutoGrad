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
  Mat x(2,2); // [[x1, x2], [1, 1]]
  x.set(0, 0, 1); // x1=1
  x.set(1, 0, 1);
  x.set(0, 1, 0); // x2=0
  x.set(1, 1, 1);
  cout << "x:\n";
  x.printMat();

  Mat label(2,1); // [[label1], [label2]]
  label.set(0, 0, 2); // label1=2
  label.set(1, 0, 1); // label2=1
  cout << "label:\n";
  label.printMat();

  Mat w(2,1); // [[w1], [1]]
  w.set(0, 0, 0.5);
  w.set(1, 0, 0.5);
  cout << "w:\n";
  w.printMat();
  w.transpose().printMat(); // [[w, b]]
  Mat y = w.transpose() * x;  // [[y1, y2]] = [[w*x1+b, w*x2+b]]
  cout << "y:\n";
  y.printMat();
  Mat diff = y.transpose() - label;
  cout << "diff:\n";
  diff.printMat();
  Mat se_loss = diff.transpose() * diff;
  cout << "se:\n";
  se_loss.printMat();
  cout << "start of backward\n";
  se_loss.backward();
  cout << "end of backward\n";
  
  x.printGradient();
  w.printGradient();

  x.clear();
  w.clear();
}

int main () {
  // test1();
  // cout << "\n";
  test2();
}