# Folk了顺便打颗星星呗【卑微脸】


# Loading data


```python
import pandas as pd
import numpy as np
```


```python
myDf=pd.read_csv("data/test.csv")
```

# Preprocessing


```python
myDf["text"]=myDf["text"].apply(lambda x:x+" <end>")
myDf["tag"]=myDf["tag"].apply(lambda x:x+" END")
myDf[:1]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>将 军 百 战 死 &lt;end&gt;</td>
      <td>B I B I S END</td>
    </tr>
  </tbody>
</table>
</div>




```python
myDf.dropna(inplace=True)
```

# Transforming data to one-hot embedding to generalize X


```python
wordIndexDict={"<pad>":0}
wi=1
for row in myDf["text"].values.tolist():
    if type(row)==float:
        print(row)
        break
    for word in row.split(" "):
        if word not in wordIndexDict:
            wordIndexDict[word]=wi
            wi+=1
vocabSize=wi
```


```python
maxLen=max(len(row) for row in myDf["text"].values.tolist())
sequenceLengths=[len(row) for row in myDf["text"].values.tolist()]
```


```python
myDf["text"]=myDf["text"].apply(lambda x:[wordIndexDict[word] for word in x.split()])
```


```python
import tensorflow as tf
X=tf.keras.preprocessing.sequence.pad_sequences(myDf["text"],
                                                value=wordIndexDict["<pad>"],
                                                padding='post',
                                                maxlen=maxLen)
X
```




    array([[ 1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 1,  2,  7,  8,  9,  4, 10,  6,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 1,  1,  2,  8,  9,  4, 11,  6,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 2,  4,  1,  2,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 1,  2,  4,  1,  2,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 1,  2,  4,  1,  2,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0]])



# Generalizing Y


```python
import tqdm
import re

myDf["tag"]=myDf["tag"].apply(lambda x:re.sub("\-\S+","",x))

tagIndexDict = {"PAD": 0}
ti = 1
for row in tqdm.tqdm(myDf["tag"].values.tolist()):
    for tag in row.split(" "):
        if tag not in tagIndexDict:
            tagIndexDict[tag] = ti
            ti += 1
tagSum = len(list(tagIndexDict.keys()))
myDf["tag"] = myDf["tag"].apply(lambda x:x.split()+["PAD" for i in range(maxLen-len(x.split()))])
myDf["tag"] = myDf["tag"].apply(lambda x:[tagIndexDict[tagItem] for tagItem in x])
# myDf["tag"] = myDf["tag"].apply(lambda x: [[0 if tagI != tagIndexDict[tagItem] else 1
#                                             for tagI in range(len(tagIndexDict))]
#                                             for tagItem in x])
y=np.array(myDf["tag"].values.tolist())
```

    100%|██████████| 6/6 [00:00<?, ?it/s]
    


```python
y.shape # it is OK whether y is one-hot embedding or not
```




    (6, 19)



# Generalizing Model


```python
from BiLSTMCRF import MyBiLSTMCRF
myModel=MyBiLSTMCRF(vocabSize,maxLen, tagIndexDict,tagSum,sequenceLengths)
```


```python
myModel.myBiLSTMCRF.summary()
```

    Model: "sequential_9"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    embedding_9 (Embedding)      (None, 19, 100)           1200
    _________________________________________________________________
    bidirectional_18 (Bidirectio (None, 19, 5)             4240
    _________________________________________________________________
    bidirectional_19 (Bidirectio (None, 19, 5)             440
    _________________________________________________________________
    crf_layer (CRF)              (None, 19)                65
    =================================================================
    Total params: 5,945
    Trainable params: 5,945
    Non-trainable params: 0
    _________________________________________________________________
    

# training model


```python
history=myModel.fit(X,y,epochs=1500)
```

    .9719
    Epoch 1263/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9693
    Epoch 1264/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9667
    Epoch 1265/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9641
    Epoch 1266/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9616
    Epoch 1267/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9590
    Epoch 1268/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9565
    Epoch 1269/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9539
    Epoch 1270/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9514
    Epoch 1271/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9489
    Epoch 1272/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9463
    Epoch 1273/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.9438
    Epoch 1274/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9413
    Epoch 1275/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9388
    Epoch 1276/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9363
    Epoch 1277/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9338
    Epoch 1278/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9313
    Epoch 1279/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9288
    Epoch 1280/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9264
    Epoch 1281/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9239
    Epoch 1282/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9214
    Epoch 1283/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9190
    Epoch 1284/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.9165
    Epoch 1285/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9141
    Epoch 1286/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.9116
    Epoch 1287/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9092
    Epoch 1288/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9068
    Epoch 1289/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9043
    Epoch 1290/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.9019
    Epoch 1291/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8995
    Epoch 1292/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8971
    Epoch 1293/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.8947
    Epoch 1294/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8923
    Epoch 1295/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8899
    Epoch 1296/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8875
    Epoch 1297/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8852
    Epoch 1298/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.8828
    Epoch 1299/1500
    6/6 [==============================] - 0s 9ms/sample - loss: 1.8804
    Epoch 1300/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8781
    Epoch 1301/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8757
    Epoch 1302/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8734
    Epoch 1303/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8710
    Epoch 1304/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8687
    Epoch 1305/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8663
    Epoch 1306/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8640
    Epoch 1307/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8617
    Epoch 1308/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8594
    Epoch 1309/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8571
    Epoch 1310/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8547
    Epoch 1311/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8525
    Epoch 1312/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8502
    Epoch 1313/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8479
    Epoch 1314/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8456
    Epoch 1315/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8433
    Epoch 1316/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8410
    Epoch 1317/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8388
    Epoch 1318/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8365
    Epoch 1319/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8342
    Epoch 1320/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.8320
    Epoch 1321/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8297
    Epoch 1322/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8275
    Epoch 1323/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.8252
    Epoch 1324/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8230
    Epoch 1325/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8208
    Epoch 1326/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.8185
    Epoch 1327/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8163
    Epoch 1328/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8141
    Epoch 1329/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8119
    Epoch 1330/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8097
    Epoch 1331/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8075
    Epoch 1332/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8053
    Epoch 1333/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.8031
    Epoch 1334/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.8009
    Epoch 1335/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7987
    Epoch 1336/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7965
    Epoch 1337/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7944
    Epoch 1338/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7922
    Epoch 1339/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7900
    Epoch 1340/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7879
    Epoch 1341/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7857
    Epoch 1342/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7836
    Epoch 1343/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7814
    Epoch 1344/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7793
    Epoch 1345/1500
    6/6 [==============================] - 0s 9ms/sample - loss: 1.7771
    Epoch 1346/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7750
    Epoch 1347/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7729
    Epoch 1348/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7707
    Epoch 1349/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7686
    Epoch 1350/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7665
    Epoch 1351/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7644
    Epoch 1352/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7623
    Epoch 1353/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7602
    Epoch 1354/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7581
    Epoch 1355/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7560
    Epoch 1356/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7539
    Epoch 1357/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7518
    Epoch 1358/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7497
    Epoch 1359/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7476
    Epoch 1360/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7456
    Epoch 1361/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7435
    Epoch 1362/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7414
    Epoch 1363/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7394
    Epoch 1364/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7373
    Epoch 1365/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7353
    Epoch 1366/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7332
    Epoch 1367/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7312
    Epoch 1368/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7291
    Epoch 1369/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7271
    Epoch 1370/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7251
    Epoch 1371/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7230
    Epoch 1372/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7210
    Epoch 1373/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7190
    Epoch 1374/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7169
    Epoch 1375/1500
    6/6 [==============================] - 0s 9ms/sample - loss: 1.7149
    Epoch 1376/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.7129
    Epoch 1377/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7109
    Epoch 1378/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7089
    Epoch 1379/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7069
    Epoch 1380/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7049
    Epoch 1381/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.7029
    Epoch 1382/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.7009
    Epoch 1383/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6989
    Epoch 1384/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6969
    Epoch 1385/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6950
    Epoch 1386/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6930
    Epoch 1387/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.6910
    Epoch 1388/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6891
    Epoch 1389/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6871
    Epoch 1390/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6851
    Epoch 1391/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.6832
    Epoch 1392/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6812
    Epoch 1393/1500
    6/6 [==============================] - 0s 11ms/sample - loss: 1.6793
    Epoch 1394/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6773
    Epoch 1395/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6754
    Epoch 1396/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.6735
    Epoch 1397/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6715
    Epoch 1398/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6696
    Epoch 1399/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6677
    Epoch 1400/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6657
    Epoch 1401/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6638
    Epoch 1402/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6619
    Epoch 1403/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6600
    Epoch 1404/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6581
    Epoch 1405/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6562
    Epoch 1406/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6543
    Epoch 1407/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6524
    Epoch 1408/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6505
    Epoch 1409/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6486
    Epoch 1410/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6467
    Epoch 1411/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6448
    Epoch 1412/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6429
    Epoch 1413/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6410
    Epoch 1414/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.6392
    Epoch 1415/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6373
    Epoch 1416/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6354
    Epoch 1417/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6336
    Epoch 1418/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6317
    Epoch 1419/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6298
    Epoch 1420/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6280
    Epoch 1421/1500
    6/6 [==============================] - 0s 11ms/sample - loss: 1.6261
    Epoch 1422/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6243
    Epoch 1423/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6224
    Epoch 1424/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6206
    Epoch 1425/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6187
    Epoch 1426/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6169
    Epoch 1427/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6150
    Epoch 1428/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.6132
    Epoch 1429/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.6114
    Epoch 1430/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6096
    Epoch 1431/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6077
    Epoch 1432/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6059
    Epoch 1433/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.6041
    Epoch 1434/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6023
    Epoch 1435/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.6005
    Epoch 1436/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5987
    Epoch 1437/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5969
    Epoch 1438/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5951
    Epoch 1439/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5933
    Epoch 1440/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5915
    Epoch 1441/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5897
    Epoch 1442/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5879
    Epoch 1443/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5861
    Epoch 1444/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5843
    Epoch 1445/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5825
    Epoch 1446/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5808
    Epoch 1447/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5790
    Epoch 1448/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5772
    Epoch 1449/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5755
    Epoch 1450/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5737
    Epoch 1451/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5719
    Epoch 1452/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5702
    Epoch 1453/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5684
    Epoch 1454/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5666
    Epoch 1455/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5649
    Epoch 1456/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5631
    Epoch 1457/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5614
    Epoch 1458/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5597
    Epoch 1459/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5579
    Epoch 1460/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5562
    Epoch 1461/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5544
    Epoch 1462/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5527
    Epoch 1463/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5510
    Epoch 1464/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5493
    Epoch 1465/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5475
    Epoch 1466/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5458
    Epoch 1467/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5441
    Epoch 1468/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5424
    Epoch 1469/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5407
    Epoch 1470/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5389
    Epoch 1471/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5372
    Epoch 1472/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5355
    Epoch 1473/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5338
    Epoch 1474/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5321
    Epoch 1475/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5304
    Epoch 1476/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5288
    Epoch 1477/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5271
    Epoch 1478/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5254
    Epoch 1479/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5237
    Epoch 1480/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5220
    Epoch 1481/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5203
    Epoch 1482/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5187
    Epoch 1483/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5170
    Epoch 1484/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5153
    Epoch 1485/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.5136
    Epoch 1486/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5120
    Epoch 1487/1500
    6/6 [==============================] - 0s 8ms/sample - loss: 1.5103
    Epoch 1488/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5087
    Epoch 1489/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5070
    Epoch 1490/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5053
    Epoch 1491/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5037
    Epoch 1492/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5020
    Epoch 1493/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.5004
    Epoch 1494/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.4987
    Epoch 1495/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.4971
    Epoch 1496/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.4955
    Epoch 1497/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.4938
    Epoch 1498/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.4922
    Epoch 1499/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.4906
    Epoch 1500/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.4889
    

# predicting


```python
testI=2
```


```python
preY=myModel.predict(X)[testI]
```


```python
indexTagDict=dict(list(zip(list(tagIndexDict.values()),list(tagIndexDict.keys()))))
indexWordDict=dict(list(zip(list(wordIndexDict.values()),list(wordIndexDict.keys()))))

sentenceList=[indexWordDict[wordItem] for wordItem in X[testI]]
sentenceList=sentenceList[:sentenceList.index("<end>")]

tagList=[indexTagDict[tagItem] for tagItem in preY]
tagList=tagList[:tagList.index("END")]

print(" ".join(sentenceList))
print(" ".join(tagList))
```

    将 将 军 带 上 战 车
    S B I B I B I
    
```
