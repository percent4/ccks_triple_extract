# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-03-13 03:08

import json

# 载入关系对应表
with open("relation2id.json", "r", encoding="utf-8") as h:
    relation_dict = json.loads(h.read())

# 加载数据集
def load_data(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.readlines()

    content = [_.replace(' ', '').replace('\u3000', '').replace('\xa0', '').replace('\u2003', '') for _ in content]

    for l in content:
        l = json.loads(l)
        D.append({
            'text': l['text'],
            'spo_list': [
                (spo['subject'], spo['predicate'], spo['object'])
                for spo in l['spo_list']
            ]
        })
    return D

# 训练集
filename = "train_data.json"
D = load_data(filename=filename)

f = open("train.txt", "w", encoding="utf-8")

for seq in D[:5]:
    subjects = list(set([_[0] for _ in seq["spo_list"]]))
    objects = list(set([_[-1] for _ in seq["spo_list"]]))

    for subj in subjects:
        for obj in objects:
            select_subj_obj = [_ for _ in seq["spo_list"] if _[0] == subj and _[-1] == obj]
            if select_subj_obj:
                f.write(str(relation_dict[select_subj_obj[0][1]])+' '+subj+'$'+obj+'$'+
                        seq["text"].replace(subj, '#'*len(subj)).replace(obj, "#"*len(obj))+'\n')
            else:
                f.write(str(0) + ' ' + subj + '$' + obj + '$' +
                        seq["text"].replace(subj, '#' * len(subj)).replace(obj, "#" * len(obj))+'\n')

f.close()

# 测试集
filename = "test_data.json"
D = load_data(filename=filename)

f = open("test.txt", "w", encoding="utf-8")

for seq in D[:1]:
    subjects = list(set([_[0] for _ in seq["spo_list"]]))
    objects = list(set([_[-1] for _ in seq["spo_list"]]))

    for subj in subjects:
        for obj in objects:
            select_subj_obj = [_ for _ in seq["spo_list"] if _[0] == subj and _[-1] == obj]
            if select_subj_obj:
                f.write(str(relation_dict[select_subj_obj[0][1]])+' '+subj+'$'+obj+'$'+
                        seq["text"].replace(subj, '#'*len(subj)).replace(obj, "#"*len(obj))+'\n')
            else:
                f.write(str(0) + ' ' + subj + '$' + obj + '$' +
                        seq["text"].replace(subj, '#' * len(subj)).replace(obj, "#" * len(obj))+'\n')

f.close()



