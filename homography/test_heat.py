import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib
import seaborn as sns
import planar

# 1340, 890
x = [200, 450, 1100, 800]
y  =[698, 38, 679, 120]
ax = sns.kdeplot(x, y)
plt.xlim(0, 1340)
plt.ylim(0, 890)
plt.show()
