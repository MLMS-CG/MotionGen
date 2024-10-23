import numpy as np
import os
import json
import matplotlib.pyplot as plt

featurePath = "/home/kxue/Work/MotionGen/AMASS/"
dataPath = "/home/kxue/Work/MotionGen/babel_v1.0_release/"

files = [
    "extra_train.json", # 7921 objs
    "extra_val.json", # 2636 objs
    "test.json", # 2084
    "train.json", # 6615
    "val.json", #2193
]

datas = []
for file in files:
    f = open(dataPath + file)
    datas.append(json.load(f))

exTrain, exVal, test, train, val = datas

print(
    "Size of train: %d, val: %d, test: %d, extra train: %d, extra val: %d" % 
    (len(train.keys()), len(val.keys()), len(test.keys()), len(exTrain.keys()), len(exVal.keys()))
)

keyExTrain = exTrain.keys()
counter = 0
for key in keyExTrain:
    if exTrain[key]["frame_anns"]:
        counter += 1
print(
    "there are %d of %d sequences in extra trainset have frame level annotation"
    % (counter, len(exTrain.keys()))
)

keyTrain = train.keys()
keyVal = val.keys()
keyTest = test.keys()

# build a dict of proc_label-raw_label
procTrans = {}
for key in keyTrain:
    objLabels = train[key]["seq_ann"]["labels"]
    for label in objLabels:
        if label["raw_label"]!=label["proc_label"]:
            if label["proc_label"] not in procTrans.keys():
                procTrans[label["proc_label"]] = set()
            procTrans[label["proc_label"]].add(label["raw_label"])
procTrans

# build a dict of action-proc_label
actProclabel = {}
for key in keyTrain:
    objLabels = train[key]["seq_ann"]["labels"]
    for label in objLabels:
        if label["act_cat"]:
            for action in label["act_cat"]:
                if action not in actProclabel.keys():
                    actProclabel[action] = set()
                actProclabel[action].add(label["proc_label"])
actProclabel