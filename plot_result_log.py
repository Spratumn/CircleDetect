import matplotlib.pyplot as plt
import pandas as pd
import os

from config import Config


cfg = Config()
train_data = pd.read_csv(os.path.join(cfg.LOG_DIR, 'train_log_teach.csv'))
val_data = pd.read_csv(os.path.join(cfg.LOG_DIR, 'val_log_teach.csv'))

total_loss = train_data['total_loss'].values[1:]
hm_loss = train_data['hm_loss'].values[1:]
wh_loss = train_data['wh_loss'].values[1:]
offset_loss = train_data['offset_loss'].values[1:]

precision = val_data['precision'].values[1:]
recall = val_data['recall'].values[1:]

plt.figure()

plt.plot(range(len(total_loss)), total_loss)
plt.legend(["total_loss_teach"])
plt.show()

plt.plot(range(len(total_loss)), hm_loss)
plt.legend(["hm_loss_teach"])
plt.show()

plt.plot(range(len(total_loss)), wh_loss)
plt.legend(["wh_loss_teach"])
plt.show()
#
plt.plot(range(len(total_loss)), offset_loss)
plt.legend(["offset_loss_teach"])
plt.show()

plt.plot(range(len(precision)), precision)
plt.legend(["precision_teach"])
plt.show()

plt.plot(range(len(recall)), recall)
plt.legend(["recall_teach"])
plt.show()
