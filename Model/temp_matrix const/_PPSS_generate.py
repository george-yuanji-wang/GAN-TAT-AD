import numpy as np
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ap = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\PPI_matrix.txt'
bp = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\STP_adj_matrix.txt'

a = np.loadtxt(ap)
b = np.loadtxt(bp)

final_matrix = np.add(a, b)
final_matrix[final_matrix > 0] = 1

print(final_matrix.sum())

np.savetxt('_PPSS.txt', final_matrix, fmt='%.3f')
base_cmap = plt.cm.spring
colors = [(0, 0, 0)] + [base_cmap(x) for x in np.linspace(0.001, np.max(final_matrix), 256)]
cmap_name = 'custom_spring'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Plot and save the heatmap
plt.imshow(final_matrix, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1)
plt.colorbar()
plt.savefig('_PPSS.png', dpi=1000)
plt.close()