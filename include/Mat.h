#ifndef MAT_H
#define MAT_H

#include "Float.h"
#include <vector>
using namespace std;

class Size {
private:
  int* data;

public:
  Size(int row, int col) {
    // cout << "construct\n";
    data = new int[2];
    data[0] = row;
    data[1] = col;
  }

  Size(const Size & s) {
    // cout << "copy construct\n";
    data = new int[2];
    data[0] = s.data[0];
    data[1] = s.data[1];
  }

  int operator[](int i) {
    if (i < 2) {
      return data[i];
    }
    return -1;
  }

  ~Size() {
    delete data;
    data = nullptr;
  }
};

class Mat {
private:
  vector< vector<_Float*> > data;
  int row = 0;
  int col = 0;

  Mat() {

  }

public:

  Mat (int row, int col, bool stop_grad=false, float default_value=0) {
    cout << typeid(*this).name() << " " << this << " is being created.\n";
    this->row = row;
    this->col = col;
    for (int i = 0; i < row; ++i) {
      vector<_Float*> rdata;
      for (int j = 0; j < col; ++j) {
        _Float * p = new _Float(default_value, stop_grad);
        rdata.push_back(p);
      }
      data.push_back(rdata);
    }
  }

  bool isIndexValid(int r, int c) const {
    return ((r >= 0 && r < row) && (c >= 0 && c < col));
  }

  bool set (int r, int c, float d) {
    if (!isIndexValid(r,c))
      return false;
    data[r][c]->data = d;
    return true;
  }

  float at (int r, int c) const {
    if (!isIndexValid(r,c))
      return 0;
    return data[r][c]->data;
  }

  Size shape() {
    return Size(row, col);
  }

  ~Mat () {
    // cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }

  void printMat() {
    cout << "[";
    for (int i = 0; i < row; ++i) {
      cout << "[";
      for (int j = 0; j < col; ++j) {
        cout << data[i][j]->data;
        if (j != col-1)
          cout << ",";
      }
      cout << "]";
      if (i != row-1)
        cout << ",\n";
    }
    printf("], shape=(%d, %d)\n", row, col);
  }

  void printGradient() {
    cout << "[";
    for (int i = 0; i < row; ++i) {
      cout << "[";
      for (int j = 0; j < col; ++j) {
        data[i][j]->printGradient();
        if (j != col-1)
          cout << ",";
      }
      cout << "]";
      if (i != row-1)
        cout << ",\n";
    }
    printf("], shape=(%d, %d)\n", row, col);
  }

  vector<vector<float>> getGradient() {
    vector<vector<float>> grad;
    for (int i = 0; i < row; ++i) {
      vector<float> rdata;
      for (int j = 0; j < col; ++j) {
        rdata.push_back(this->data[i][j]->grad->data);
      }
      grad.push_back(rdata);
    }
    return grad;
  }

  void backward() {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        data[i][j]->backward();
      }
    }
  }

  void update(float learning_rate, vector<vector<float>> updates) { // 手动指定梯度更新量
    if (updates.size() != row || updates[0].size() != col)
      throw -1;
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        float new_data = data[i][j]->data - updates[i][j] * learning_rate;
        data[i][j]->data = new_data;
      }
    }
  }

  void update(float learning_rate) {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        float new_data = data[i][j]->data - data[i][j]->grad->data * learning_rate;
        data[i][j]->data = new_data;
      }
    }
  }

  void clear() {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        delete data[i][j];
      }
    }
  }

  Mat transpose() const {
    int row2 = this->col;
    int col2 = this->row;
    Mat mat2;
    mat2.row = row2;
    mat2.col = col2;
    for (int i = 0; i < row2; ++i) {
      vector<_Float*> rdata;
      for (int j = 0; j < col2; ++j) {
        rdata.push_back(this->data[j][i]);
      }
      mat2.data.push_back(rdata);
    }
    return mat2;
  }

  friend Mat operator+(const Mat& mat1, const Mat & mat2); // 矩阵相加
  friend Mat operator-(const Mat& mat1, const Mat & mat2); // 矩阵相减
  friend Mat operator*(const Mat& mat1, const Mat & mat2); // 矩阵相乘
  friend Mat operator*(const Mat& mat1, const float a); // 矩阵与标量相乘
  friend Mat operator*(const float a, const Mat& mat1); // 标量与矩阵相乘
  friend Mat operator/(const Mat& mat1, const float a); // 矩阵每个元素各自除以标量
  friend Mat operator/(const Mat& mat1, const Mat& mat2); // 矩阵对应元素相除或广播（只实现了一种可广播的情况）
  friend Mat Hadamard(const Mat& mat1, const Mat& mat2); // Hadamard Product 对应元素相乘
  friend Mat sigmoid(const Mat&mat1); // 逐元素取sigmoid
  friend Mat ReLU(const Mat&mat1); // 逐元素取ReLU
  friend Mat LogSigmoid(const Mat&mat1); // 逐元素取LogSigmoid
  friend Mat operator-(const Mat& mat1); // 逐元素取负号
  friend Mat sum(const Mat& mat1, int dim); // 沿dim轴取和
  friend Mat BCEWithLogitsLoss(const Mat& input, const Mat& target, const Mat& ones);

  Mat max(int dim=0) {
    if (dim == 0) {
      Mat mat2(1, col, true);
      float max[col];
      for (int j = 0; j < col; ++j) {
        max[j] = this->data[0][j]->data;
      }
      for (int i = 1; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
          float cur = this->data[i][j]->data;
          if (cur > max[j]) {
            max[j] = cur;
          }
        }
      }
      for (int j = 0; j < col; ++j) {
        mat2.set(0, j, max[j]);
      }
      return mat2;
    }
    else if (dim == 1) {
      Mat mat2(row, 1, true);
      float max[row];
      for (int i = 0; i < row; ++i) {
        max[i] = this->data[i][0]->data;
      }
      for (int j = 1; j < col; ++j) {
        for (int i = 0; i < row; ++i) {
          float cur = this->data[i][j]->data;
          if (cur > max[i]) {
            max[i] = cur;
          }
        }
      }
      for (int i = 0; i < row; ++i) {
        mat2.set(i, 0, max[i]);
      }
      return mat2;
    }
    else {
      throw -1;
    }
  }

  class BatchNorm {
  private:
    int features;
    float miu;
    const float eps = 0.0000000001;

  public:
    // 可学习
    Mat* betas; // [1, features] 均值
    Mat* gammas; // [1, features] 方差

    // 从训练数据统计
    Mat* running_means; // [1, features] 均值
    Mat* running_vars; // [1, features] 方差

    // 每一次前向运算都不同，反向传播完后要清除
    vector<_Float*> tmp_means;
    vector<_Float*> tmp_vars;

    BatchNorm(int features, float momentum=0.9) { // 作用的数据是features维
      betas = new Mat(1, features, false, 0);
      gammas = new Mat(1, features, false, 1);
      this->features = features;
      this->miu = momentum;

      running_means = new Mat(1, features, true, 0);
      running_vars = new Mat(1, features, true, 1);
    }

    Mat operator()(Mat& x, int mode) { // [batch, features], mode: 0 for train, 1 for test
      if (x.col != features)
        throw -1;
      int batch = x.row;
      
      Mat x_norm; // 未分配数据
      x_norm.row = batch;
      x_norm.col = features;

      tmp_means.clear();
      tmp_vars.clear();
      if (mode == 0) { // 训练阶段
        for (int f = 0; f < features; ++f) {
          float tmp_mean = 0;
          for (int b = 0; b < batch; ++b) {
            tmp_mean += x.at(b, f);
          }
          tmp_mean /= batch;

          float tmp_var = 0;
          for (int b = 0; b < batch; ++b) {
            tmp_var += pow(x.at(b, f) - tmp_mean, 2);
          }
          tmp_var /= batch;

          // 利用batch均值方差做归一化normalize
          // for (int b = 0; b < batch; ++b) {
          //   x.data[b][f]->data = (x.at(b, f) - tmp_mean) / sqrt(tmp_var + eps);
          // }
          tmp_means.push_back(new _Float(tmp_mean, true));
          tmp_vars.push_back(new _Float(sqrt(tmp_var + eps), true)); // 注意存了整个分母
          
          //计算 running_means和running_vars 供测试用
          running_means->set(0, f, miu*running_means->at(0, f) + (1-miu)*tmp_mean);
          running_vars->set(0, f, miu*running_vars->at(0, f) + (1-miu)*tmp_var);
        }
        for (int i = 0; i < batch; ++i) {
          vector<_Float*> rdata;
          for (int j = 0; j < features; ++j) {
            _Float *p = &((*x.data[i][j] - *tmp_means[j]) / *(tmp_vars[j]));
            rdata.push_back(p);
          }
          x_norm.data.push_back(rdata);
        }
        // scale and shift
        Mat y = Hadamard(x_norm, (*gammas)) + (*betas); // H([batch, features], [1, features]) + [1, features]
        return y;
      }
      // else if (mode == 1) { // 测试阶段
      //   for (int f = 0; f < features; ++f) {
      //     for (int b = 0; b < batch; ++b) {
      //       x.data[b][f]->data = (x.at(b, f) - running_means->at(0, f)) / sqrt(running_vars->at(0, f) + eps);
      //     }
      //   }
      //   Mat y = Hadamard(x, (*gammas)) + (*betas);
      //   return y;
      // }
      else {
        string info = "BatchNorm: mode not implemented.";
        throw info;
      }
    }

    void clear() {
      gammas->clear();
      betas->clear();
      delete gammas;
      delete betas;
      gammas = nullptr;
      betas = nullptr;
    }

    // 反向传播后调用，否则内存泄漏
    void clear_tmp_means_vars() {
      for (int i = 0; i < tmp_means.size(); ++i) {
        delete tmp_means[i];
        delete tmp_vars[i];
      }
      tmp_means.clear();
      tmp_vars.clear();
    }

    ~BatchNorm() {

    }
  };
};

Mat operator+(const Mat& mat1, const Mat& mat2) {
  try {
    if (mat1.row == mat2.row && mat1.col == mat2.col) { // 维度相等，对应元素相加
      int row = mat1.row;
      int col = mat2.col;
      Mat mat3; // 没有分配数据
      mat3.row = row;
      mat3.col = col;
      for (int i = 0; i < row; ++i) {
        vector<_Float*> rdata;
        for (int j = 0; j < col; ++j) {
          _Float *p = &(*(mat1.data[i][j]) + *(mat2.data[i][j]));
          rdata.push_back(p);
        }
        mat3.data.push_back(rdata);
      }
      return mat3;
    }
    else if (mat2.row == 1 && mat2.col == mat1.col) { // 广播加法 [row, col] + [1, col]
      int row = mat1.row;
      int col = mat1.col;
      Mat mat3; // 没有分配数据
      mat3.row = row;
      mat3.col = col;
      for (int i = 0; i < row; ++i) {
        vector<_Float*> rdata;
        for (int j = 0; j < col; ++j) {
          _Float *p = &(*(mat1.data[i][j]) + *(mat2.data[0][j])); // 区别所在
          rdata.push_back(p);
        }
        mat3.data.push_back(rdata);
      }
      return mat3;
    }
    else if (mat2.col == 1 && mat2.row == mat1.row) { // 广播加法 [row, col] + [row, 1]
      return (mat1.transpose() + mat2.transpose()).transpose(); // 转置成列相等，相加后再转置回来
    }
    else {
      string info = "Mat operator + error: matrices dimension incompatible";
      throw info;
    }
  }
  catch (string info) {
    cout << info << "\n";
  }
}

Mat Hadamard(const Mat& mat1, const Mat& mat2) { // 对应元素相乘
  if (mat1.row == mat2.row && mat1.col == mat2.col) {
    int row = mat1.row;
    int col = mat2.col;
    Mat mat3; // 没有分配数据
    mat3.row = row;
    mat3.col = col;
    for (int i = 0; i < row; ++i) {
      vector<_Float*> rdata;
      for (int j = 0; j < col; ++j) {
        _Float *p = &(*(mat1.data[i][j]) * *(mat2.data[i][j]));
        rdata.push_back(p);
      }
      mat3.data.push_back(rdata);
    }
    return mat3;
  }
  if (mat2.col == 1 && mat2.row == mat1.row) { // [row, col] and [row, 1]
    int row = mat1.row;
    int col = mat1.col;
    Mat mat3; // 没有分配数据
    mat3.row = row;
    mat3.col = col;
    for (int i = 0; i < row; ++i) {
      vector<_Float*> rdata;
      for (int j = 0; j < col; ++j) {
        _Float *p = &(*(mat1.data[i][j]) * *(mat2.data[i][0]));
        rdata.push_back(p);
      }
      mat3.data.push_back(rdata);
    }
    return mat3;
  }
  else if (mat2.row == 1 && mat2.col == mat1.col) { // [row, col] and [1, col]
    return Hadamard(mat1.transpose(), mat2.transpose()).transpose(); // -> [col, row] and [col, 1] = [col, row] -> [row, col]
  }
  else {
    cout << "erro\n";
    string info = "Hadamard(mat1, mat2): incompatible dimensions";
    throw info;
  }
}

Mat operator-(const Mat& mat1, const Mat& mat2) {
  if (mat1.row != mat2.row || mat1.col != mat2.col)
    throw -1;
  int row = mat1.row;
  int col = mat2.col;
  Mat mat3; // 没有分配数据
  mat3.row = row;
  mat3.col = col;
  for (int i = 0; i < row; ++i) {
    vector<_Float*> rdata;
    for (int j = 0; j < col; ++j) {
      _Float *p = &(*(mat1.data[i][j]) - *(mat2.data[i][j]));
      rdata.push_back(p);
    }
    mat3.data.push_back(rdata);
  }
  return mat3;
}

Mat operator*(const Mat& mat1, const Mat &mat2) { // 矩阵乘法
  if (mat1.col != mat2.row)
    throw -1;
  int row = mat1.row;
  int col = mat2.col;
  int range = mat1.col;
  Mat mat3;
  mat3.row = row;
  mat3.col = col;
  for (int i = 0; i < row; ++i) {
    vector<_Float*> rdata;
    for (int j = 0; j < col; ++j) {
      _Float *p = &(*(mat1.data[i][0]) * *(mat2.data[0][j]));
      for (int k = 1; k < range; ++k) {
        p = &(*mat1.data[i][k] * *(mat2.data[k][j]) + *p);
      }
      rdata.push_back(p);
    }
    mat3.data.push_back(rdata);
  }
  return mat3;
}

Mat operator/(const Mat& mat1, const Mat& mat2) {
  try {
    if (mat1.row == mat2.row && mat1.col == mat2.col) { // 维度相等，对应元素相除
      int row = mat1.row;
      int col = mat1.col;
      Mat mat3;
      mat3.row = row;
      mat3.col = col;
      for (int i = 0; i < row; ++i) {
        vector<_Float*> rdata;
        for (int j = 0; j < col; ++j) {
          _Float *p = &(*(mat1.data[i][j]) / *(mat2.data[i][j]));
          rdata.push_back(p);
        }
        mat3.data.push_back(rdata);
      }
      return mat3;
    }
    else if (mat2.row == 1 && mat2.col == mat1.col) { // 可广播的一种情况: [row,col] / [1,col]
      int row = mat1.row;
      int col = mat1.col;
      Mat mat3;
      mat3.row = row;
      mat3.col = col;
      for (int i = 0; i < row; ++i) {
        vector<_Float*> rdata;
        for (int j = 0; j < col; ++j) {
          _Float *p = &(*(mat1.data[i][j]) / *(mat2.data[0][j]));
          rdata.push_back(p);
        }
        mat3.data.push_back(rdata);
      }
      return mat3;
    }
    else {
      string info = "Mat operator / error: matrices dimension incompatible";
      throw info;
    }
  } catch (string info) {
    cout << info << "\n";
  }
}

Mat operator*(const Mat& mat1, const float a) {
  _Float *ap = new _Float(a, true); // 内存泄漏
  Mat mat2;
  mat2.row = mat1.row;
  mat2.col = mat1.col;
  for (int i = 0; i < mat2.row; ++i) {
    vector<_Float*> rdata;
    for (int j = 0; j < mat2.col; ++j) {
      _Float *p = &(*mat1.data[i][j] * *ap);
      rdata.push_back(p);
    }
    mat2.data.push_back(rdata);
  }
  return mat2;
}

Mat operator*(const float a, const Mat& mat1) {
  return mat1 * a;
}

Mat operator/(const Mat& mat1, const float a) {
  if (a == 0)
    throw -1;
  return mat1 * (1.0/a); // 内存泄漏
}

Mat sigmoid(const Mat& mat1) {
  Mat mat2;
  mat2.row = mat1.row;
  mat2.col = mat1.col;
  for (int i = 0; i < mat2.row; ++i) {
    vector<_Float*> rdata;
    for (int j = 0; j < mat2.col; ++j) {
      _Float *p = &(sigmoid(*mat1.data[i][j]));
      rdata.push_back(p);
    }
    mat2.data.push_back(rdata);
  }
  return mat2;
}

Mat ReLU(const Mat& mat1) {
  Mat mat2;
  mat2.row = mat1.row;
  mat2.col = mat1.col;
  for (int i = 0 ; i < mat2.row; ++i) {
    vector<_Float*> rdata;
    for (int j = 0; j < mat2.col; ++j) {
      _Float *p = &(ReLU(*mat1.data[i][j]));
      rdata.push_back(p);
    }
    mat2.data.push_back(rdata);
  }
  return mat2;
}

Mat LogSigmoid(const Mat& mat1) {
  Mat mat2;
  mat2.row = mat1.row;
  mat2.col = mat1.col;
  for (int i = 0 ; i < mat2.row; ++i) {
    vector<_Float*> rdata;
    for (int j = 0; j < mat2.col; ++j) {
      _Float *p = &(LogSigmoid(*mat1.data[i][j]));
      rdata.push_back(p);
    }
    mat2.data.push_back(rdata);
  }
  return mat2;
}

Mat operator-(const Mat& mat1) {
  Mat mat2;
  mat2.row = mat1.row;
  mat2.col = mat1.col;
  for (int i = 0 ; i < mat2.row; ++i) {
    vector<_Float*> rdata;
    for (int j = 0; j < mat2.col; ++j) {
      _Float *p = &(-(*mat1.data[i][j]));
      rdata.push_back(p);
    }
    mat2.data.push_back(rdata);
  }
  return mat2;
}

Mat sum(const Mat& mat1, int dim=0) {
  if (dim == 1) {
    Mat mat2;
    mat2.row = mat1.row;
    mat2.col = 1;
    for (int i = 0; i < mat1.row; ++i) {
      vector<_Float*> rdata;
      _Float *tmp_sum = mat1.data[i][0];
      for (int j = 1; j < mat1.col; ++j) {
        tmp_sum = &(*tmp_sum + *mat1.data[i][j]);
      }
      rdata.push_back(tmp_sum);
      mat2.data.push_back(rdata);
    }
    return mat2;
  }
  else if (dim == 0) {
    Mat mat2 = sum(mat1.transpose(), 1).transpose();
    return mat2;
  }
  else {
    throw -1;
  }
}

Mat BCEWithLogitsLoss(const Mat& input, const Mat& target, const Mat& ones) { // shape = [Batch, 1]
  Mat logLikelihood = (Hadamard(target, LogSigmoid(input)) + Hadamard(ones-target, LogSigmoid(-input))); // shape [B, 1] // log(1-sigmoid(x)) = log(sigmoid(-x))
  Mat loss = -sum(logLikelihood, 0);
  return loss;
}

#endif