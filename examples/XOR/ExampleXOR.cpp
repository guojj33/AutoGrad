#include "Mat.h"
#include "Optim.h"
#include <fstream>

void randInit(Mat& mat) {
  for (int i = 0; i < mat.shape()[0]; ++i) {
    for (int j = 0; j < mat.shape()[1]; ++j) {
      mat.set(i, j, rand() / double(RAND_MAX));
    }
  }
}

int main() {
  const int count = 4;
  float x1[count] = {0, 0, 1, 1};
  float x2[count] = {0, 1, 0, 1};
  float y[count] = {0, 1, 1, 0}; // y = x1 xor x2
  // float y[count] = {0, 0, 1, 1}; // y = x1

  int input_dim = 2;
  int output_dim = 1;
  Mat X(count, input_dim, true);
  Mat Y(count, output_dim, true);
  for (int i = 0; i < count; ++i) {
    X.set(i, 0, x1[i]);
    X.set(i, 1, x2[i]);
    Y.set(i, 0, y[i]);
  }
  X.printMat();
  Y.printMat();

  int hidden_dim = 2;
  Mat W(input_dim, hidden_dim, false);
  Mat c(hidden_dim, 1, false);
  Mat w(hidden_dim, output_dim, false);
  Mat b(output_dim, 1, false);
  Mat::BatchNorm bn0(input_dim, 1.0);
  Mat::BatchNorm bn1(hidden_dim, 1.0);

  vector<Mat> params;
  params.push_back(W);
  params.push_back(c);
  params.push_back(w);
  params.push_back(b);
  for (int i = 0; i < params.size(); ++i) {
    randInit(params[i]);
    params[i].printMat();
  }
  // params.push_back(*bn0.gammas);
  // params.push_back(*bn0.betas);
  // params.push_back(*bn1.gammas);
  // params.push_back(*bn1.betas);

  /*
  可行配置1：
  隐藏层sigmoid激活
  隐藏层h归一化
  epoch = 4000
  lr = 0.00005
  momentum = 0.9
  在第1250次迭代lr乘0.1

  可行配置2：
  隐藏层sigmoid激活
  输入层和隐藏层h都归一化
  epoch = 2000
  lr = 0.001
  momentum = 0.9
  不需手动改lr
  
  make
  ./bin/ExampleXOR.out > output.log & python plot.py 
  */

  float lr = 0.001;
  float momentum = 0.9;
  Momentum optimizer(params, lr, momentum); // 可能实现错了
  optimizer.reset();

  int epoch = 2000;
  vector<float> trainLoss;

  // 手动定义是为了最后能手动释放空间
  Mat ones(count, 1, true, 1); // 与Y的维度相同，用于参与BCEWithLogitsoss的计算
  Mat num(1, 1, true, count); // 作为分子，计算损失的平均

  for (int e = 0; e < epoch; ++e) {
    Mat X_norm = bn0(X, 0);
    Mat h = sigmoid(X_norm*W+c.transpose()); // ([4,2]*[2,2] + [1,2]) = [4,2]
    Mat h_norm = bn1(h, 0); // [4,2]
    Mat y_pred = h_norm*w + b.transpose(); // [4,2]*[2,1] + [1,1] = [4,1]
    Mat loss = BCEWithLogitsLoss(y_pred, Y, ones); // [1,1]
    Mat mean_loss = loss / num;
    float loss_item = mean_loss.at(0, 0);
    trainLoss.push_back(loss_item);
    
    if ((e+1) % 1 == 0) {
        printf("\nEPOCH %d/%d:\n", e+1, epoch);
        printf("loss:\n");
        mean_loss.printMat();
        printf("y_pred:\n");
        y_pred.printMat();
    }

    mean_loss.backward();
    optimizer.step();
  }
  
  printf("异号: %d, 同号: %d", optimizer.tmpCount1, optimizer.tmpCount2);
  fstream fout("./output/train_loss.txt", ios::out);
  for (int i = 0; i < trainLoss.size(); ++i) {
    fout << trainLoss[i] << "\n";
  }
  fout.close();

  // 打印网络参数
  cout << "bn0:\n";
  cout << "  means:\n";
  bn0.running_means->printMat();
  cout << " vars:\n";
  bn0.running_vars->printMat();
  cout << " gammas:\n";
  bn0.gammas->printMat();
  cout << " betas:\n";
  bn0.betas->printMat();

  cout << "bn1:\n";
  cout << "  means:\n";
  bn1.running_means->printMat();
  cout << " vars:\n";
  bn1.running_vars->printMat();
  cout << " gammas:\n";
  bn1.gammas->printMat();
  cout << " betas:\n";
  bn1.betas->printMat();

  cout << "W:\n";
  W.printMat();
  cout << "c:\n";
  c.printMat();
  cout << "w:\n";
  w.printMat();
  cout << "b:\n";
  b.printMat();

  X.clear();
  Y.clear();
  W.clear();
  c.clear();
  w.clear();
  b.clear();
  bn0.clear();
  bn1.clear();

  ones.clear();
  num.clear();
}