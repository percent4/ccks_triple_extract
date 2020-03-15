# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-03-13 03:03
import os, re, json, traceback
from pprint import pprint

with open("all_50_schemas", "r", encoding="utf-8") as f:
    content = [_.strip() for _ in f.readlines()]

relation_dict = {"未知": 0}

for i, line in enumerate(content):
    relation_dict[json.loads(line)["predicate"]] = i+1

with open("relation2id.json", "w", encoding="utf-8") as g:
    g.write(json.dumps(relation_dict, ensure_ascii=False, indent=2))