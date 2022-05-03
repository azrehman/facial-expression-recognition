import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

matrix = [[61, 0,  1,  0,  0,  3,  5],
         [ 1, 65,  2,  1,  1,  0,  0],
         [ 0,  0, 69,  0,  0,  1,  0],
         [ 0,  0,  0, 70,  0,  0,  0],
         [ 0,  0,  1,  0, 69,  0,  0],
         [ 0,  1,  1,  0,  3, 65,  0],
         [ 0,  0,  0,  0,  0,  0, 70]]

df = pd.DataFrame(matrix, range(7), range(7))

sn.set(font_scale=1.4)
cmap = sn.light_palette('seagreen', as_cmap=True)
sn.heatmap(df, annot=True, annot_kws={'size': 16}, cmap=cmap)
plt.show()
