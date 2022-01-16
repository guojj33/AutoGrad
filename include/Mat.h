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
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        cout << data[i][j]->data << " ";
      }
      cout << "\n";
    }
  }

  void printGradient() {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        data[i][j]->printGradient();
        cout << " ";
      }
      cout << "\n";
    }
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

  friend Mat operator+(const Mat& mat1, const Mat & mat2);
  friend Mat operator-(const Mat& mat1, const Mat & mat2);
  friend Mat operator*(const Mat& mat1, const Mat & mat2);
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