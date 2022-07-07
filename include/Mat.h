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

  Mat (int row, int col, bool stop_grad=false) {
    cout << typeid(*this).name() << " " << this << " is being created.\n";
    this->row = row;
    this->col = col;
    for (int i = 0; i < row; ++i) {
      vector<_Float*> rdata;
      for (int j = 0; j < col; ++j) {
        _Float * p = new _Float(stop_grad);
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

  void backward() {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        data[i][j]->backward();
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

  Mat transpose() {
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
  friend Mat sigmoid(const Mat&mat1); // 逐元素取sigmoid

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
};

Mat operator+(const Mat& mat1, const Mat& mat2) {
  try {
    if (mat1.row == mat2.row && mat1.col != mat2.col) {
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
    else {
      string info = "Mat operator + error: matrices dimension incompatible";
      throw info;
    }
  }
  catch (string info) {
    cout << info << "\n";
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

Mat operator*(const Mat& mat1, const Mat &mat2) {
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
  _Float *ap = new _Float(a, true);
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
  return mat1 * (1.0/a);
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

#endif