# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-03-11 13:16
import json
import numpy as np
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import load_model
from collections import defaultdict
from pprint import pprint

from sequence_labeling.utils import MAX_SEQ_LEN, event_type
from albert_zh.extract_feature import BertVector

# 读取label2id字典
with open("%s_label2id.json" % event_type, "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v: k for k, v in label_id_dict.items()}

# 利用ALBERT提取文本特征
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)
f = lambda text: bert_model.encode([text])["encodes"][0]

# 载入模型
custom_objects = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy}
ner_model = load_model("%s_ner.h5" % event_type, custom_objects=custom_objects)


# 从预测的标签列表中获取实体
def get_entity(sent, tags_list):

    entity_dict = defaultdict(list)
    i = 0
    for char, tag in zip(sent, tags_list):
        if 'B-' in tag:
            entity = char
            j = i+1
            entity_type = tag.split('-')[-1]
            while j < min(len(sent), len(tags_list)) and 'I-%s' % entity_type in tags_list[j]:
                entity += sent[j]
                j += 1

            entity_dict[entity_type].append(entity)

        i += 1

    return dict(entity_dict)


# 输入句子，进行预测
while 1:
    # 输入句子
    text = input("Please enter an sentence: ").replace(' ', '')
    # 利用训练好的模型进行预测
    train_x = np.array([f(text)])
    y = np.argmax(ner_model.predict(train_x), axis=2)
    y = [id_label_dict[_] for _ in y[0] if _]

    # 输出预测结果
    pprint(get_entity(text, y))

