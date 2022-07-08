#include "Float.h"
#include "Mat.h"
#include <iostream>
using namespace std;

// Float 测试
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

// Mat 测试
void test2() {
  Mat x(2, 2, true); // [[x1, x2], [1, 1]]
  x.set(0, 0, 1); // x1=1
  x.set(1, 0, 1);
  x.set(0, 1, 0); // x2=0
  x.set(1, 1, 1);
  cout << "x:\n";
  x.printMat();
  x.printGradient();

  Mat label(2, 1, true); // [[label1], [label2]]
  label.set(0, 0, 2); // label1=2
  label.set(1, 0, 1); // label2=1
  cout << "label:\n";
  label.printMat();

  Mat w(2,1); // [[w1], [1]]
  w.set(0, 0, 0.5);
  w.set(1, 0, 0.5);
  float learing_rate = 0.01;

  for (int epoch = 0; epoch < 10; ++epoch) {
    printf("--------------epoch %d-------------\n", epoch);
    cout << "w:\n";
    w.transpose().printMat(); // [[w, b]]
    Mat y = w.transpose() * x;  // [[y1, y2]] = [[w*x1+b, w*x2+b]]
    cout << "y:\n";
    y.printMat();
    Mat diff = y.transpose() - label;
    Mat se_loss = diff.transpose() * diff;
    se_loss = se_loss/2;
    cout << "se_loss:\n";
    se_loss.printMat();
    cout << "start of backward\n";
    se_loss.backward();
    cout << "end of backward\n";
    
    // w.printGradient();
    w.update(learing_rate);
  }

  x.clear();
  w.clear();
}

// 除法测试
void test3() {
  Float f1(2);
  Float f2(1);
  Float f3 = f1/f2;
  f3.backward();
  f1.printGradient();
  f2.printGradient();
  f1.clear();
  f2.clear();
}

// max测试
void test4() {
  Mat mat(4, 3);
  int count = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      mat.set(i, j, count + i - j);
    }
  }
  mat.printMat();
  Mat max0 = mat.max(0);
  max0.printMat();
  max0.clear();
  Mat max1 = mat.max(1);
  max1.printMat();
  max1.clear();
}