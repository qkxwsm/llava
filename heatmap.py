import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt("full_attention_heads.tsv", delimiter='\t')
print(data.shape)
plt.imshow(data, cmap='seismic', interpolation='nearest', vmin=0, vmax=1)

# Add a colorbar
cbar = plt.colorbar()

# Set bin edges and labels
bin_edges = np.arange(1, 33, 1) / 32
#bin_labels = [f"{i/10:.1f}-{j/10:.1f}" for i, j in zip(range(10), range(1, 11))]

xticks = np.arange(0, data.shape[1] + 0, 2)
yticks = np.arange(0, data.shape[0] + 0, 2)
plt.xticks(ticks=xticks, labels=np.arange(1, data.shape[1] + 1, 2))
plt.yticks(ticks=yticks, labels=np.arange(1, data.shape[0] + 1, 2))



data = data.flatten()
data.sort()

uq = data[int(len(data)*3/4-1)]
cbar.ax.axhline(uq, c='black')

m = data[int(len(data)/2-1)]
cbar.ax.axhline(m, c='black')

labels = [(x, str(round(x,2))) for x in np.arange(0.0, 1.2, 0.2)]
labels.append((m, 'Mean'))
labels.append((uq, 'Upper\nQuartile'))
labels.sort(key=lambda x: x[0])
cbar.ax.set_yticks([x[0] for x in labels])
cbar.ax.set_yticklabels([x[1] for x in labels])


plt.xlabel('Attention Head')
plt.ylabel('Layer')
plt.title('Learned Gate Values for LLaVA')
plt.savefig("gateheat.png")


