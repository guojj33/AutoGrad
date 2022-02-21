#ifndef MAT_H
#define MAT_H

#include "Float.h"
#include <vector>
using namespace std;

class Mat {
private:
  vector< vector<_Float*> > data;
  int row = 0;
  int col = 0;

  Mat() {

  }

public:

  Mat (int row, int col) {
    cout << typeid(*this).name() << " " << this << " is being created.\n";
    this->row = row;
    this->col = col;
    for (int i = 0; i < row; ++i) {
      vector<_Float*> rdata;
      for (int j = 0; j < col; ++j) {
        _Float * p = new _Float();
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

  ~Mat () {
    cout << typeid(*this).name() << " " << this << " is being deleted.\n";
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
    cout << "]\n";
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
    cout << "]\n";
  }

  void backward() {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        data[i][j]->backward();
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

  friend Mat operator+(const Mat& mat1, const Mat & mat2);
  friend Mat operator-(const Mat& mat1, const Mat & mat2);
  friend Mat operator*(const Mat& mat1, const Mat & mat2);
  friend Mat operator*(const Mat& mat1, const float a);
  friend Mat operator*(const float a, const Mat& mat1);
};

Mat operator+(const Mat& mat1, const Mat& mat2) {
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
      _Float *p = &(*(mat1.data[i][j]) + *(mat2.data[i][j]));
      rdata.push_back(p);
    }
    mat3.data.push_back(rdata);
  }
  return mat3;
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


Mat operator*(const Mat& mat1, const float a) {

}

Mat operator*(const float a, const Mat& mat1) {

}

#endif