# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-03-12 21:52
import json
from pprint import pprint
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 8), dpi=100)   # 输出图片大小为1800*800
# Mac系统设置中文字体支持
plt.rcParams["font.family"] = 'Arial Unicode MS'


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

filename = '../data/train_data.json'

D = load_data(filename=filename)
pprint(D)

# 创建text, text_length, spo_num的DataFrame
text_list = [_["text"] for _ in D]
spo_num = [len(_["spo_list"])for _ in D]

df = pd.DataFrame({"text": text_list, "spo_num": spo_num} )
df["text_length"] = df["text"].apply(lambda x: len(x))
print(df.head())
print(df.describe())

# 绘制spo_num的条形统计图
pprint(df['spo_num'].value_counts())
label_list = list(df['spo_num'].value_counts().index)
num_list = df['spo_num'].value_counts().tolist()

# 利用Matplotlib模块绘制条形图
x = range(len(num_list))
rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="频数")
plt.ylim(0, 80000) # y轴范围
plt.ylabel("数量")
plt.xticks([index + 0.1 for index in x], label_list)
plt.xlabel("三元组数量")
plt.title("三元组频数统计图")

# 条形图的文字说明
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

# plt.show()
plt.savefig('./spo_num_bar_chart.png')

plt.close()

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 8), dpi=100)   # 输出图片大小为1800*800
# Mac系统设置中文字体支持
plt.rcParams["font.family"] = 'Arial Unicode MS'


# 关系统计图
relation_dict = defaultdict(int)

for spo_dict in D:
    # print(spo_dict["spo_list"])
    for spo in spo_dict["spo_list"]:
        relation_dict[spo[1]] += 1

label_list = list(relation_dict.keys())
num_list = list(relation_dict.values())

# 利用Matplotlib模块绘制条形图
x = range(len(num_list))
rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="频数")
# plt.ylim(0, 80000) # y轴范围
plt.ylabel("数量")
plt.xticks([index + 0.1 for index in x], label_list)
plt.xticks(rotation=45) # x轴的标签旋转45度
plt.xlabel("三元组关系")
plt.title("三元组关系频数统计图")

# 条形图的文字说明
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")


plt.savefig('./relation_bar_chart.png')

