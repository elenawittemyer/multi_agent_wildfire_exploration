import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iter_1 = np.array([178.6, 174, 170.4, 152, 166.4, 168, 151, 163, 155.2])
iter_2 = np.array([142.8, 153.2, 143.6, 167.8, 149.8, 146.2, 133, 159.2, 185.2])
iter_3 = np.array([133.2, 146.8, 155.4, 152.2, 135.6, 172.2, 154.8, 142.6, 113.2])

static = pd.DataFrame({'reduc_1': [25.586, 40.849, 33.863, 36.163, 38.981, 41.82, 36.288, 45.188, 47.466, 23.68],
                       'reduc_2': [37.159, 58.577, 45.795, 43.766, 55.529, 44.972, 46.918, 52.313, 48.749, 32.827],
                       'reduc_3': [28.617, 53.692, 37.846, 44.603, 48.081, 48.375, 38.333, 44.219, 46.194, 43.787],
                       'label': ['static', 'static', 'static', 'static', 'static', 'static', 'static', 'static', 'static', 'static']})



static['z_score'] = static['reduc_1'].apply(lambda x: (x - static['reduc_1'].mean()) / static['reduc_1'].std())


labels = ['Baseline', 'Smoke Mask', 'Shannon Entropy']
font_title = {'fontname':'Times New Roman', 'size': 16}
font_ax = {'fontname':'Times New Roman', 'size': 14}
font_subax = {'fontname':'Times New Roman', 'size': 12}
fig, ax = plt.subplots()
bplot = ax.boxplot([reduc_1, reduc_2, reduc_3], notch=False, patch_artist=True)

for patch in bplot['boxes']:
    patch.set_facecolor('teal')
for median in bplot['medians']:
    median.set_color('black')

ax.set_xticklabels(labels, **font_subax)
ax.set_xlabel('Expected Informationb   Measure', **font_ax)
ax.set_ylabel('Uncertainty Reduction (%)', **font_ax)
ax.set_title('Information Measure vs Uncertainty \n Map Reduction', **font_title)
plt.show()