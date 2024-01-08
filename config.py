import os
import numpy as np
import matplotlib.pyplot as plt
# path_std = 'save/stds/'
# path_c0 = 'save/stds/' 
# path_c1 = 'save/stds/' 
# path_c2 = 'save/stds/'
# filenames = os.listdir(path_std) 
# for file in filenames:
#     std = np.load(path_std + file)
#     c0 = np.load(path_std + file)
#     c1 = np.load(path_std + file)
#     c2 = np.load(path_std + file)
#     fig, (ax0,ax1,ax2,ax3) = plt.subplots(ncols=1, nrows=4)
#     ax0.imshow(std, origin='lower', cmap='Greys')
#     ax1.imshow(c0, origin='lower', cmap='Greys')
#     ax2.imshow(c1, origin='lower', cmap='Greys')
#     ax3.imshow(c2, origin='lower', cmap='Greys')
#     plt.savefig('save/imgs/' + file.split('.')[0] + '.png')
#     # print(arr.shape)

# from scipy.interpolate import griddata
# labels = np.load('save/labels.npy')
# positions = np.load('save/positions.npy')

# x, y = np.meshgrid(np.linspace(-10,10,1000), np.linspace(-10,10,1000))
# print(x)
# z = griddata(positions[:-1], labels[1:], (x, y), method='nearest')
# print(z.shape)

# plt.imshow(z, extent=[-10,10,-10,10], origin='lower')
# plt.scatter(positions.T[0], positions.T[1], c='k')
# plt.savefig('save/test.png')

# data = np.load('save/testimg.npy')
# plt.imshow(data.T, origin='lower')
# plt.savefig('save/testimg.png')

# pcs = np.load('save/pca/pc50.npy')
# print(pcs.shape)

# # plt.imshow(pcs[0].reshape((320,240)))
# fig, axs = plt.subplots(1,5)
# print(axs.flatten().shape)
# for pc, ax in zip(pcs, axs.flatten()):
#     ax.imshow(pc.reshape((400,400)),origin='lower')
# fig.tight_layout()
# plt.savefig('save/pc_pic.png')

# np.save('id_counter.npy', np.array([0]))

