# Loading data


import pandas as pd
import numpy as np
import tqdm
import re
from BiLSTMCRF import MyBiLSTMCRF
import tensorflow as tf


def load_data():
    myDf = pd.read_csv("data/test.csv")

    # Preprocessing

    myDf["text"] = myDf["text"].apply(lambda x: x + " <end>")
    myDf["tag"] = myDf["tag"].apply(lambda x: x + " END")
    print(myDf[:1])

    myDf.dropna(inplace=True)

    # Transforming data to one-hot embedding to generalize X

    wordIndexDict = {"<pad>": 0}
    wi = 1
    for row in myDf["text"].values.tolist():
        if type(row) == float:
            print(row)
            break
        for word in row.split(" "):
            if word not in wordIndexDict:
                wordIndexDict[word] = wi
                wi += 1
    vocabSize = wi

    maxLen = max(len(row) for row in myDf["text"].values.tolist())
    sequenceLengths = [len(row) for row in myDf["text"].values.tolist()]

    myDf["text"] = myDf["text"].apply(lambda x: [wordIndexDict[word] for word in x.split()])

    X = tf.keras.preprocessing.sequence.pad_sequences(myDf["text"],
                                                      value=wordIndexDict["<pad>"],
                                                      padding='post',
                                                      maxlen=maxLen)
    print(X)

    # Generalizing Y

    myDf["tag"] = myDf["tag"].apply(lambda x: re.sub("\-\S+", "", x))
    tagIndexDict = {"PAD": 0}
    ti = 1
    for row in tqdm.tqdm(myDf["tag"].values.tolist()):
        for tag in row.split(" "):
            if tag not in tagIndexDict:
                tagIndexDict[tag] = ti
                ti += 1
    tagSum = len(list(tagIndexDict.keys()))
    myDf["tag"] = myDf["tag"].apply(lambda x: x.split() + ["PAD" for i in range(maxLen - len(x.split()))])
    myDf["tag"] = myDf["tag"].apply(lambda x: [tagIndexDict[tagItem] for tagItem in x])
    # myDf["tag"] = myDf["tag"].apply(lambda x: [[0 if tagI != tagIndexDict[tagItem] else 1
    #                                             for tagI in range(len(tagIndexDict))]
    #                                             for tagItem in x])
    y = np.array(myDf["tag"].values.tolist())

    print(y.shape)  # it is OK whether y is one-hot embedding or not
    return X, y, vocabSize, maxLen, tagIndexDict, wordIndexDict, tagSum, sequenceLengths


# Generalizing Model
def train():
    X, y, vocabSize, maxLen, tagIndexDict, wordIndexDict, tagSum, sequenceLengths = load_data()
    myModel = MyBiLSTMCRF(vocabSize, maxLen, tagIndexDict, tagSum, sequenceLengths)
    myModel.myBiLSTMCRF.summary()
    # training model
    history = myModel.fit(X, y, epochs=1500)
    # predicting
    testI = 2
    preY = myModel.predict(X)[testI]
    indexTagDict = dict(list(zip(list(tagIndexDict.values()), list(tagIndexDict.keys()))))
    indexWordDict = dict(list(zip(list(wordIndexDict.values()), list(wordIndexDict.keys()))))

    sentenceList = [indexWordDict[wordItem] for wordItem in X[testI]]
    sentenceList = sentenceList[:sentenceList.index("<end>")]

    tagList = [indexTagDict[tagItem] for tagItem in preY]
    tagList = tagList[:tagList.index("END")]

    print(" ".join(sentenceList))
    print(" ".join(tagList))


if __name__ == '__main__':
    train()
