import torch
import numpy as np
import matplotlib.pyplot as plt

opt_train_lh = torch.load('../data/opt_train_lh.pt').detach().cpu().numpy()
opt_test_lh = torch.load('../data/opt_test_lh.pt').detach().cpu().numpy()
reg_train_lh = torch.load('../data/reg_train_lh.pt').detach().cpu().numpy()
reg_test_lh = torch.load('../data/reg_test_lh.pt').detach().cpu().numpy()

plt.figure(figsize=(7, 4))
plt.semilogy(reg_train_lh, label='Regular')
plt.semilogy(opt_train_lh, label='Optimized')
plt.grid()
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig('train_lh.pdf')

plt.figure(figsize=(7, 4))
plt.semilogy(reg_test_lh, label='Regular')
plt.semilogy(opt_test_lh, label='Optimized')
plt.grid()
plt.title('Validation Loss History')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig('test_lh.pdf')
