# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-03-11 10:04
import json

from sequence_labeling.utils import train_file_path, event_type


# 读取数据集
def read_data(file_path):
    # 读取数据集
    with open(file_path, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    # 添加原文句子以及该句子的标签

    # 读取空行所在的行号
    index = [-1]
    index.extend([i for i, _ in enumerate(content) if ' ' not in _])
    index.append(len(content))

    # 按空行分割，读取原文句子及标注序列
    sentences, tags = [], []
    for j in range(len(index)-1):
        sent, tag = [], []
        segment = content[index[j]+1: index[j+1]]
        for line in segment:
            sent.append(line.split()[0])
            tag.append(line.split()[-1])

        sentences.append(''.join(sent))
        tags.append(tag)

    # 去除空的句子及标注序列，一般放在末尾
    sentences = [_ for _ in sentences if _]
    tags = [_ for _ in tags if _]

    return sentences[:50000], tags[:50000]


# 读取训练集数据
# 将标签转换成id
def label2id():

    train_sents, train_tags = read_data(train_file_path)

    # 标签转换成id，并保存成文件
    unique_tags = []
    for seq in train_tags:
        for _ in seq:
            if _ not in unique_tags:
                unique_tags.append(_)

    label_id_dict = dict(zip(unique_tags, range(1, len(unique_tags) + 1)))

    with open("%s_label2id.json" % event_type, "w", encoding="utf-8") as g:
        g.write(json.dumps(label_id_dict, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    label2id()