#include "Mat.h"

int main() {
  const int count = 4;
  float x1[count] = {0, 0, 1, 1};
  float x2[count] = {0, 1, 0, 1};
  float y[count] = {0, 1, 1, 0};

  Mat X(4, 2, true);
  Mat Y(4, 1, true);
  for (int i = 0; i < 4; ++i) {
    X.set(i, 0, x1[i]);
    X.set(i, 1, x2[i]);
    Y.set(i, 0, y[i]);
  }
  X.printMat();
  Y.printMat();

  Mat W(2, 2);
  Mat c(2, 1);
  Mat w(2, 1);
  Mat b(1, 1);

  int epoch = 100;
  float lr_rate = 0.001;
  vector<float> trainLoss;

  for (int e = 0; e < epoch; ++e) {
    Mat y_pred = w.transpose() * ReLU((X*W).transpose() + c) + b;
    Mat diff = y_pred - Y;
    Mat se_loss = diff.transpose() * diff;
    se_loss = se_loss / count;
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
  }
}