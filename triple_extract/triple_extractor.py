# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-03-14 20:41
import os, re, json, traceback

import json
import numpy as np
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import load_model
from collections import defaultdict
from pprint import pprint
from text_classification.att import Attention

from albert_zh.extract_feature import BertVector

# 读取label2id字典
with open("../sequence_labeling/ccks2019_label2id.json", "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v: k for k, v in label_id_dict.items()}
# 利用ALBERT提取文本特征
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=128)
f = lambda text: bert_model.encode([text])["encodes"][0]

# 载入NER模型
custom_objects = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy}
ner_model = load_model("../sequence_labeling/ccks2019_ner.h5", custom_objects=custom_objects)

# 载入分类模型
best_model_path = '../text_classification/models/per-rel-19-0.9329.h5'
classification_model = load_model(best_model_path, custom_objects={"Attention": Attention})

# 分类与id的对应关系
with open("../data/relation2id.json", "r", encoding="utf-8") as g:
    relation_id_dict = json.loads(g.read())

id_relation_dict = {v: k for k, v in relation_id_dict.items()}


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

class TripleExtract(object):

    def __init__(self, text):
        self.text = text.replace(" ", "")    # 输入句子

    # 获取输入句子中的实体（即：主体和客体）
    def get_entity(self):
        train_x = np.array([f(self. text)])
        y = np.argmax(ner_model.predict(train_x), axis=2)
        y = [id_label_dict[_] for _ in y[0] if _]

        # 输出预测结果
        return get_entity(self.text, y)

    # 对实体做关系判定
    def relation_classify(self):
        entities = self.get_entity()
        subjects = list(set(entities.get("SUBJ", [])))
        objs = list(set(entities.get("OBJ", [])))

        spo_list = []

        for subj in subjects:
            for obj in objs:
                sample = '$'.join([subj, obj, self.text.replace(subj, '#'*len(subj)).replace(obj, "#"*len(obj))])
                vec = bert_model.encode([sample])["encodes"][0]
                x_train = np.array([vec])

                # 模型预测并输出预测结果
                predicted = classification_model.predict(x_train)
                y = np.argmax(predicted[0])

                relation = id_relation_dict[y]
                if relation != "未知":
                    spo_list.append([subj, relation, obj])

        return spo_list

    # 提取三元组
    def extractor(self):

        return self.relation_classify()


