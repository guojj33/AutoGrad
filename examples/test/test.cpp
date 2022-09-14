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

// pow测试
void test5() {
  Float x(2);
  Float y = pow(x, 3);
  Float w(3);
  Float l = w*y; // l = 3*(x^3)
  l.backward();
  x.printGradient(); // 36
}

// relu测试
void test6() {
  Float x(2);
  Float w(3);
  Float y = w*x; // y = 3*x
  Float l = ReLU(y); // l = max(0, 3*x)
  l.backward();
  x.printGradient(); // 3
}

void test7() {
  Mat x(2,1);
  x.set(0, 0, 2);
  x.set(1, 0, -1);
  Mat y = ReLU(x*2);
  y.printMat();
  y.backward();
  x.printGradient();
}

void test8() {
  Mat mat1(2,2,true,2);
  Mat mat2(1,2,true,1);
  mat2.set(0,0,0.5);
  Mat mat3 = mat1 + mat2;
  mat3.printMat();
  mat3.backward();
  mat1.clear();
  mat2.clear();
}

// logsigmoid test
void test9() {
  Mat mat1(1,2,false,1);
  mat1.set(0,0,0.5);
  Mat mat2 = LogSigmoid(mat1);
  mat1.printMat();
  mat2.printMat();
  mat2.backward();
  mat1.printGradient();
  mat1.clear();
}

// negative test
void test10() {
  Mat mat1(1,2,false,4);
  Mat mat2 = -mat1*2;
  mat2.printMat();
  mat2.backward();
  mat1.printGradient();
  mat1.clear();
}

// sum test
void test11() {
  Mat mat1(2,2,false,4);
  mat1.set(0,0,1);
  mat1.printMat();
  Mat mat2 = sum(mat1, 1);
  mat2.printMat();
  mat2.backward();
  mat1.printGradient();
  mat1.clear();
}

#include <random>
void test12() {
  default_random_engine gen;
  normal_distribution<float> dist(5, 4); // mean std_deviaiton

  int features = 1;
  int batch = 1000;
  Mat x(batch, features, false, 1);
  for (int i = 0; i < batch; ++i) {
    x.set(i, 0, dist(gen));
  }
  // x.printMat();
  float momentum = 0;
  Mat::BatchNorm bn(features, momentum);
  // x.printMat();
  Mat y = bn(x, 0);
  // y.printMat();
  float m = bn.running_means->at(0,0);
  float v = bn.running_vars->at(0, 0);
  cout << m << " " << v << "\n"; // 5.12411 16.916
  Mat:: BatchNorm bn1(features, momentum);
  Mat y1 = bn(y, 0);
  
  m = bn.running_means->at(0,0);
  v = bn.running_vars->at(0, 0);
  cout << m << " " << v << "\n"; // 0 1

  y.backward();
  y1.clear();
  x.clear();
  bn.clear();
  bn1.clear();
}

int main() {
  // test1();
  // test2();
  // test3();
  // test4();
  // test5();
  // test6();
  // test7();
  // test8();
  // test9();
  // test10();
  // test11();
  test12();
}