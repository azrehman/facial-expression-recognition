import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


train_hist_df = pd.read_csv('resnet50/FEC_resnet50_trained_history.csv')

fig, axes = plt.subplots(nrows=2, ncols=1)

train_hist_df.plot(y=['train_loss', 'val_loss'], ax=axes[0], title='ResNet50 Loss', xlabel='Epochs', ylabel='Loss')
axes[0].legend(['train', 'val'])
axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))

train_hist_df.plot(y=['train_acc', 'val_acc'], ax=axes[1], title='ResNet50 Accuracy', xlabel='Epochs', ylabel='Accuracy')
axes[1].legend(['train', 'val'])
axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

fig.tight_layout()

plt.show()

