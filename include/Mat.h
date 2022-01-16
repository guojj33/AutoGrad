#include "_Float.h"
#include <vector>
using namespace std;

class Mat {
private:
  vector< vector<_Float*> > data;
  int row = 0;
  int col = 0;

public:
  Mat() {

  }

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

  void set (int r, int c, float d) {
    data[r][c]->data = d;
  }

  float at (int r, int c) const {
    return data[r][c]->data;
  }

  ~Mat () {
    cout << typeid(*this).name() << " " << this << " is being deleted.\n";
  }

  friend Mat operator+(const Mat& mat1, const Mat & mat2);
  friend Mat operator-(const Mat& mat1, const Mat & mat2);
  friend Mat operator*(const Mat& mat1, const Mat & mat2);

  void printMat() {
    if (data.size() == 0) // 没有分配数据
      return;
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        cout << data[i][j]->data << " ";
      }
      cout << "\n";
    }
  }
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