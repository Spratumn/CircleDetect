import matplotlib.pyplot as plt
import pandas as pd
import os

from config import Config


cfg = Config()
train_data = pd.read_csv(os.path.join(cfg.LOG_DIR, 'train_log_res_18.csv'))

total_loss = train_data['total_loss'].values[1:]
hm_loss = train_data['hm_loss'].values[1:]
wh_loss = train_data['wh_loss'].values[1:]
offset_loss = train_data['offset_loss'].values[1:]

plt.figure()
plt.title('train result:')
# plt.plot(range(len(total_loss)), total_loss)
# plt.legend(["total_loss"])
# plt.show()

# plt.plot(range(len(total_loss)), hm_loss)
# plt.legend(["hm_loss"])
# plt.show()

# plt.plot(range(len(total_loss)), wh_loss)
# plt.legend(["wh_loss"])
# plt.show()
#
plt.plot(range(len(total_loss)), offset_loss)
plt.legend(["offset_loss"])
plt.show()
