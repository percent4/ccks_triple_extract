# -*- coding: utf-8 -*-
import json

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


# 加工成BIO标注序列
def bio_sent(sent, spo_list):
    bio_list = ['O'] * len(sent)
    for item in spo_list:
        subj = item[0]
        obj = item[-1]

        for i in range(0, len(sent)-len(subj)+1):
            if sent[i:i+len(subj)] == subj:
                bio_list[i] = 'B-SUBJ'
                for j in range(1, len(subj)):
                    bio_list[i+j] = 'I-SUBJ'

        for i in range(0, len(sent)-len(obj)+1):
            if sent[i:i+len(obj)] == obj:
                bio_list[i] = 'B-OBJ'
                for j in range(1, len(obj)):
                    bio_list[i+j] = 'I-OBJ'

    return sent, bio_list


train_data = load_data('train_data.json')
with open('ccks2019.train', 'w', encoding='utf-8') as f:
    for item in train_data:
        # print(item)
        sent, bio_list = bio_sent(item['text'], item['spo_list'])
        for char, tag in zip(sent, bio_list):
            if not char:
                print(sent)
            f.write(char+' '+tag+'\n')
        f.write('\n')

test_data = load_data('test_data.json')
with open('ccks2019.test', 'w', encoding='utf-8') as f:
    for item in test_data:
        # print(item)
        sent, bio_list = bio_sent(item['text'], item['spo_list'])
        for char, tag in zip(sent, bio_list):
            if not char:
                print(sent)
            f.write(char+' '+tag+'\n')
        f.write('\n')


