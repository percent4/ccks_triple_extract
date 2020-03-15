# -*- coding: utf-8 -*-
# 对样本进行ALBERT编码

import pickle
import numpy as np
from tqdm import tqdm

from text_classification.load_data import get_train_test_pd
from albert_zh.extract_feature import BertVector


# 读取文件并进行转换
train_df, test_df = get_train_test_pd()
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=128)
print('Begin ALBERT encoding')
f = lambda text: bert_model.encode([text])["encodes"][0]

# 训练集
x_train = []
text_list = train_df['text'].tolist()
pbar = tqdm(text_list)
for bar, i in zip(pbar, range(len(text_list))):
    pbar.set_description("Processing train data")
    x_train.append(f(text_list[i]))

# np.savetxt("x_train.txt", x_train, fmt="%.4f", delimiter=',')

y_train = np.array([label for label in train_df['label']])
x_train = np.array(x_train)
# np.savez("train_x_y", x_train=x_train, y_train=y_train)

# 测试集
x_test = []
text_list = test_df['text'].tolist()
qbar = tqdm(text_list)
for bar, i in zip(qbar, range(len(text_list))):
    qbar.set_description("Processing test data")
    x_test.append(f(text_list[i]))

print('End ALBERT encoding')

y_test = np.array([label for label in test_df['label']])

# with open("x_test.pk", 'wb') as g:
#     pickle.dump(x_test, g)
# with open("y_test.pk", 'wb') as g:
#     pickle.dump(y_test, g)

x_test = np.array(x_test)
# np.savez("test_x_y", x_test=x_test, y_test=y_test)