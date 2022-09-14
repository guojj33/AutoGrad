import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

output_dir = './output/'

train_loss = pd.read_csv(output_dir + 'train_loss.txt', header=None).to_numpy()
x = range(train_loss.shape[0])
plt.plot(x, train_loss)
plt.title('train losses')
plt.xlabel("epoch")
plt.ylabel("BCE loss")
plt.savefig(output_dir + 'train_loss.png')