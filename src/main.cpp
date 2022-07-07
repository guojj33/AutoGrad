#include "Float.h"
#include "Mat.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <float.h>

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

// 波士顿房价预测 线性回归
vector<string> parseLine(string line) {
  vector<string> splits;
  string tmp = "";
  for (int i = 0; i < line.length(); ++i) {
    if (line[i] != ',') {
      tmp += line[i];
    }
    else {
      splits.push_back(tmp);
      tmp = "";
    }
  }
  splits.push_back(tmp);
  return splits;
}

void parseBoston(vector<vector<float>> &X, vector<vector<float>> &Y) {
  fstream fin("./data/boston.csv");
  string line;
  vector< vector<string> > rawData;
  while(fin.peek() != EOF) {
    fin >> line;
    rawData.push_back(parseLine(line));
  }
  fin.close();
  vector<string> headers = rawData[0];

  // vector< vector<float> > X;
  // vector< vector<float> > Y;
  X.clear();
  Y.clear();

  stringstream ss;
  for (int i = 1; i < rawData.size(); ++i) {
    vector<string> sampleStr = rawData[i];
    vector<float> sample;
    float f;
    for (int j = 0; j < sampleStr.size(); ++j) {
      ss.clear(); // 重要
      ss << sampleStr[j];
      ss >> f;
      sample.push_back(f);
    }
    vector<float> y;
    y.push_back(sample[sample.size()-1]);
    sample.pop_back();
    vector<float> x = sample;

    X.push_back(x);
    Y.push_back(y);
  }
}

void mainBoston() {
  // 加载数据
  vector<vector<float>> X;
  vector<vector<float>> Y;
  parseBoston(X, Y);

  printf("Boston dataset loaded:\n");
  printf("X shape: [%ld, %ld]\n", X.size(), X[0].size()); // [507, 13]
  printf("Y shape: [%ld, %ld]\n", Y.size(), Y[0].size()); // [507, 1]

  int trainNum = X.size()*0.8;
  // trainNum = 2; // debug
  int testNum = X.size()-trainNum;
  printf("train split: %d, test split: %d\n", trainNum, testNum); // 405, 102

  Mat trainX(trainNum, 13, true); // 多加一列用于偏置项
  Mat trainY(trainNum, 1, true);
  for (int i = 0; i < trainNum; ++i) {
    for (int j = 0; j < 13; ++j) {
      trainX.set(i, j, X[i][j]);
    }
    // trainX.set(i, 13, 1.0);
    trainY.set(i, 0, Y[i][0]);
  }
  
  Mat testX(testNum, 13, true);
  Mat testY(testNum, 1, true);
  for (int i = 0; i < testNum; ++i) {
    for (int j = 0; j < 13; ++j) {
      testX.set(i, j, X[trainNum+i][j]);
    }
    // testX.set(i, 13, 1.0);
    testY.set(i, 0, Y[trainNum+i][0]);
  }

  // trainX.printMat();
  // trainY.printMat();
  // testX.printMat();
  // testY.printMat();
  // 加载结束

  Mat W(13, 1);
  Mat b(1, 1);
  for (int i = 0; i < 13; ++i) {
    W.set(i, 0, 0.5);
  }

  int epoch = 100;
  float lr_rate = 0.001;

  Mat maximumX = trainX.max(0);
  printf("maximumX:\n");
  maximumX.printMat();
  Mat maximumY = trainY.max(0);
  printf("maximumY:\n");
  maximumY.printMat();

  // 训练
  vector<float> trainLoss;
  float last_loss = FLT_MAX;
  for (int e = 0; e < epoch; ++e) {
    Mat trainX_norm = trainX / maximumX; // 广播除法
    // Mat trainY_norm = trainY / maximumY;

    // 计算误差
    Mat y_pred = trainX_norm * W + b;
    Mat diff = y_pred - trainY;
    Mat se_loss = diff.transpose() * diff;
    se_loss = se_loss / trainNum;
    float loss_item = se_loss.at(0, 0);
    trainLoss.push_back(loss_item);
    
    if ((e+1) % 1 == 0) {
      printf("\nEPOCH %d/%d:\n", e+1, epoch);
      printf("loss:\n");
      se_loss.printMat();
    }
    if ((e+1) % 50 == 0) {
      lr_rate /= 10;
    }

    // 反向传播
    se_loss.backward();
    // 更新参数
    W.update(lr_rate);
    b.update(lr_rate);
  }
  // 测试
  Mat testX_norm = testX / maximumX;
  Mat y_pred = testX_norm * W + b;
  Mat diff = y_pred - testY;
  Mat se_loss = diff.transpose() * diff;
  se_loss = se_loss / testNum;
  printf("\ntest loss:\n");
  se_loss.printMat();
  printf("\ntest result:\n");
  y_pred.printMat();
  testY.printMat();

  // 保存数据
  fstream f("./output/train_loss.txt", ios::out);
  for (int i = 0; i < trainLoss.size(); ++i) {
    f << trainLoss[i] << "\n";
  }
  f.close();
  f.open("./output/test_y.txt", ios::out);
  for (int i = 0; i < testNum; ++i) {
    f << testY.at(i, 0) << "\n";
  }
  f.close();
  f.open("./output/pred_y.txt", ios::out);
  for (int i = 0; i < testNum; ++i) {
    f << y_pred.at(i, 0) << "\n";
  }
  f.close();

  // 释放空间
  se_loss.backward();
  trainX.clear();
  trainY.clear();
  testX.clear();
  testY.clear();
  W.clear();
}

int main () {
  // test1();
  // test2();
  // test3();
  // test4();
  mainBoston();
}