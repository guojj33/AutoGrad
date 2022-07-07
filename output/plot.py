import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

test_y = pd.read_csv('./test_y.txt', header=None).to_numpy()
pred_y = pd.read_csv('./pred_y.txt', header=None).to_numpy()
# print(pred_y)
x = range(pred_y.shape[0])
plt.figure(dpi=300)
plt.xlabel('sample')
plt.ylabel('price')
plt.scatter(x, test_y, label='ground_truths')
plt.scatter(x, pred_y, label='infer_results')
plt.legend()
plt.title('Boston housing dataset\n102 samples for testing - test results')
plt.savefig("./test.png")

plt.cla()
plt.clf()

train_loss = pd.read_csv('./train_loss.txt', header=None).to_numpy()
x = range(train_loss.shape[0])
plt.plot(x, train_loss)
plt.title('Boston housing dataset\n405 samples for training - train losses')
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.savefig('./train_loss.png')