# -*- coding: utf-8 -*-
# 模型训练

import os, json
from operator import itemgetter
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from text_classification.att import Attention
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

from text_classification.encode_data import x_train, x_test, y_train, y_test

# train_npz = np.load('train_x_y.npz')
# x_train = train_npz["x_train"]
# y_train = train_npz["y_train"]
#
# test_npz = np.load('test_x_y.npz')
# x_test = test_npz["x_test"]
# y_test = test_npz["y_test"]

# 使用第一张与第三张GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7,8"

# 将类型y值转化为ont-hot向量
num_classes = 51
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 模型结构：ALBERT + 双向GRU + Attention + FC
inputs = Input(shape=(128, 312, ))
gru = Bidirectional(GRU(128, dropout=0.2, return_sequences=True))(inputs)
attention = Attention(32)(gru)
output = Dense(num_classes, activation='softmax')(attention)
model = Model(inputs, output)

# 模型可视化
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')

# 如果原来models文件夹下存在.h5文件，则全部删除
model_dir = './models'
if os.listdir(model_dir):
    for file in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, file))

# 保存最新的val_acc最好的模型文件
filepath="models/per-rel-{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

# 模型训练以及评估
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=30, callbacks=[early_stopping, checkpoint])

print('在测试集上的效果：', model.evaluate(x_test, y_test))

with open('../data/relation2id.json', 'r', encoding='utf-8') as f:
    label_id_dict = json.loads(f.read())

sorted_label_id_dict = sorted(label_id_dict.items(), key=itemgetter(1))
print(sorted_label_id_dict)
values = [_[0] for _ in sorted_label_id_dict]
print(values)
predictions = model.predict(x_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=values))

# 绘制loss和acc图像
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='acc')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig("loss_acc.png")
