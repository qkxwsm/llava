import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt("full_attention_heads.tsv", delimiter='\t')
print(data.shape)
data = data.flatten()
data.sort()

x_position = data[int(len(data)*3/4-1)]
plt.hist(data, bins=20)
plt.axvline(x=x_position, ymax=0.9, color='black', linestyle='--')
plt.text(x_position , 380, 'Upper\nQuartile', color='black', horizontalalignment='center')
x_position = data[int(len(data)/2-1)]
plt.axvline(x=x_position, ymax=0.9, color='black', linestyle='--')
plt.text(x_position, 380, 'Median', color='black', horizontalalignment='center')

plt.xlabel('Attention Head Gate Value')
plt.ylabel('Count')
plt.title('Learned Gate Values for LLaVA')
plt.savefig("gatehist.png")


