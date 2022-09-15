import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

output_dir = './output/'

train_loss = pd.read_csv(output_dir + 'train_loss.txt', header=None).to_numpy()
x = range(train_loss.shape[0])
plt.plot(x, train_loss, )
title = 'train losses'
plt.title(title)
plt.xlabel("epoch")
plt.ylabel("BCE loss")

# train_loss = pd.read_csv(output_dir + 'train_loss not learnable.txt', header=None).to_numpy()
# plt.plot(x, train_loss, c='r')

# plt.legend(('learnable', 'not learnable'))

plt.savefig(output_dir + title + '.png')