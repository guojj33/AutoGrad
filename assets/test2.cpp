#include <iostream>
using namespace std;

int main () {
  int r = 1, c = 2;
  int **a = new int*[r];
  for (int i = 0; i < r; ++i) {
    a[i] = new int[c];
    a[i][c] = i;
    cout << a[i][c] << "\n";
  }
  for (int i = 0; i < c; ++i) {
    delete a[i];
  }
  delete a;
}