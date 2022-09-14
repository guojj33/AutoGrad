#ifndef OPTIM_H
#define OPTIM_H

#include "Mat.h"

class Momentum {
public:
  vector<Mat> params;
  vector< vector<vector<float>> > lastUpdate; // 保存动量
  float lr;
  float miu;
  int t;
  int tmpCount1 = 0;
  int tmpCount2 = 0;

  Momentum(vector<Mat> parameters, float lr=0.1, float momentum=0.9) {
    this->params = parameters;
    this->lr = lr;
    this->miu = momentum;
    this->reset();
  }

  void step() {
    this->t += 1;
    if (t > 1) {
      // 本次的梯度与上一次的更新量加权和，作为本次的更新量
      for (int i = 0; i < params.size(); ++i) {
        vector<vector<float>> curGrad = params[i].getGradient();
        vector<vector<float>> curUpdate;
        int row = params[i].shape()[0];
        int col = params[i].shape()[1];
        for (int r = 0; r < row; ++r) {
          vector<float> rdata;
          for (int c = 0; c < col; ++c) {
            float tmp = miu*lastUpdate[i][r][c] + (1-miu)*curGrad[r][c];
            if (lastUpdate[i][r][c] * curGrad[r][c] < 0) { // 异号
              tmpCount1 += 1;
            }
            else if (lastUpdate[i][r][c] * curGrad[r][c] > 0) { // 同号
              tmpCount2 += 1;
            }
            rdata.push_back(tmp);
          }
          curUpdate.push_back(rdata);
        }
        params[i].update(lr, curUpdate);
        lastUpdate[i] = curUpdate; // 更新参数后保存本次的更新量
      }
    }
    else { // 如果是没有上一次的更新量，则以本次梯度作为本次更新量
      for (int i = 0; i < params.size(); ++i) {
        vector<vector<float>> curGrad = params[i].getGradient();
        params[i].update(lr, curGrad);
        lastUpdate.push_back(curGrad); // 更新参数后保存本次的更新量
      }
    }
  }

  void reset() {
    this->t = 0;
    this->lastUpdate.clear();
  }
};

#endif