## Relative packages
import numpy as np
import os
import json

featurePath = "/home/kxue/Work/MotionGen/HumanML3D/"
dataPath = "/home/kxue/Work/MotionGen/HumanML3D/babel_v1.0_release/"

files = [
    "extra_train.json", # 7921 objs
    "extra_val.json", # 2636 objs
    "test.json", # 2084
    "train.json", # 6615
    "val.json", #2193
]


def main():
    datas = []
    for file in files:
        f = open(dataPath + file)
        datas.append(json.load(f))

    exTrain, exVal, test, train, val = datas
    # extra dataset
    # only few of annotations in extra set has frame-level labels
    # in extraTrain, about 40/7921

    # origi dataset
    keyTrain = train.keys()
    keyVal = val.keys()
    keyTest = test.keys()
    
    # folders = set()
    # for key in keyVal:
    #     folders.add(val[key]["feat_p"].split('/',-1)[0])


    # train set analysis
    # check if there exits the case in which raw_label is different with proc_label
    procTrans = {}
    for key in keyTrain:
        objLabels = train[key]["seq_ann"]["labels"]
        for label in objLabels:
            if label["raw_label"]!=label["proc_label"]:
                if label["proc_label"] not in procTrans.keys():
                    procTrans[label["proc_label"]] = []
                procTrans[label["proc_label"]].append(label["raw_label"])
          
    # build shape-proc_label dict
    shapeActDict = {}
    actShapeDict = {}

    def buildAtlas(data):
        featureData = np.load(featurePath + data["feat_p"])
        shape = tuple(featureData.f.betas)
        actCat = []
        # some sequence may only have sequence annotation
        if data["frame_ann"]:
            labels = data["frame_ann"]["labels"]
        else:
            labels = data["seq_ann"]["labels"]
        for label in labels:
            if label["act_cat"]:
                actCat = actCat + label["act_cat"]

        if shape not in shapeActDict.keys():
            shapeActDict[shape] = set()
        for act in actCat:
            shapeActDict[shape].add(act)
            if act not in actShapeDict.keys():
                actShapeDict[act] = set()
            actShapeDict[act].add(shape)

    for key in keyTrain:
        data = train[key]
        buildAtlas(data)

    for key in keyTest:
        data = test[key]
        buildAtlas(data)

    for key in keyVal:
        data = val[key]
        buildAtlas(data)
        
    actKey = actShapeDict.keys()
    shapeNum = []
    for act in actKey:
        shapeNum.append(len(actShapeDict[act]))
    sortedIndex = np.argsort(shapeNum)

    shapeKey = shapeActDict.keys()
    actNum = []
    for act in shapeKey:
        actNum.append(len(shapeActDict[act]))

    # scatter plot person(X)-action(Y)
    x, y = [], []
    sortedAct = np.array(list(actKey))[sortedIndex][::-1]
    # from person with most actions
    personIndex = np.argsort(actNum)[::-1]
    for idxX, idx in enumerate(personIndex):
        actions = shapeActDict[list(shapeKey)[idx]]
        for action in actions:
            idxY = np.where(sortedAct==action)[0][0]
            x.append(idxX)
            y.append(idxY)


    print(1)

if __name__=="__main__":
    main()