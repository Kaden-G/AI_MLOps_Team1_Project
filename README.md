# AI_MLOps_Team1_Project

To run, I've been using: `uvicorn main:app --reload`

An example input to the training endpoint:
```
{
  "dataset_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/data/initial",
  "test_size": 0.2,
  "batch_size": 8,
  "learning_rate": 0.001,
  "momentum": 0.9,
  "num_epochs": 10,
  "save_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_1",
  "model_type": "cnn",
  "confusion_matrix_save_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/test_1_cm.png"
}
```

Test 2 model:
```
{
  "dataset_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/data/initial",
  "test_size": 0.2,
  "batch_size": 8,
  "learning_rate": 0.001,
  "momentum": 0.9,
  "num_epochs": 20,
  "save_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2",
  "model_type": "cnn",
  "confusion_matrix_save_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/test_2_cm.png"
}
```


Docker:
I've been using:
`docker build . -f Dockerfile -t mlops_teamproject:latest` and `docker run -p 8000:8000 mlops_teamproject:latest`, although the docker run will require attached volumes for data/models


# Training Logs

## Test2
```
EPOCH 0 ----------------------------
TRAINING---
Average Training Loss for Epoch 0: 0.08271690366938883
Training Accuracy for Epoch 0: 0.5898969769477844
Training Recall for Epoch 0: 0.6297675967216492
Training Precision for Epoch 0: 0.9030780792236328
VALIDATION---
Average Validation Loss for Epoch 0: 0.08548393100500107
Validation Accuracy for Epoch 0: 0.5949535369873047
Validation Recall for Epoch 0: 0.5949535369873047
Validation Precision for Epoch 0: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 1 ----------------------------
TRAINING---
Average Training Loss for Epoch 1: 0.07667639222145872
Training Accuracy for Epoch 1: 0.641575276851654
Training Recall for Epoch 1: 0.67393958568573
Training Precision for Epoch 1: 0.9303614497184753
VALIDATION---
Average Validation Loss for Epoch 1: 0.13217732310295105
Validation Accuracy for Epoch 1: 0.6447543501853943
Validation Recall for Epoch 1: 0.6516778469085693
Validation Precision for Epoch 1: 0.9837892651557922
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 2 ----------------------------
TRAINING---
Average Training Loss for Epoch 2: 0.07029458980706492
Training Accuracy for Epoch 2: 0.7110335826873779
Training Recall for Epoch 2: 0.7851375937461853
Training Precision for Epoch 2: 0.8828141093254089
VALIDATION---
Average Validation Loss for Epoch 2: 1.0850597620010376
Validation Accuracy for Epoch 2: 0.7244356274604797
Validation Recall for Epoch 2: 0.9016528725624084
Validation Precision for Epoch 2: 0.7865897417068481
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 3 ----------------------------
TRAINING---
Average Training Loss for Epoch 3: 0.1564859257668663
Training Accuracy for Epoch 3: 0.5596543550491333
Training Recall for Epoch 3: 0.6932893991470337
Training Precision for Epoch 3: 0.7438162565231323
VALIDATION---
Average Validation Loss for Epoch 3: 0.0907517820596695
Validation Accuracy for Epoch 3: 0.5949535369873047
Validation Recall for Epoch 3: 0.5949535369873047
Validation Precision for Epoch 3: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 4 ----------------------------
TRAINING---
Average Training Loss for Epoch 4: 0.07741524189730425
Training Accuracy for Epoch 4: 0.6129943132400513
Training Recall for Epoch 4: 0.8472668528556824
Training Precision for Epoch 4: 0.6891462802886963
VALIDATION---
Average Validation Loss for Epoch 4: 0.22640080749988556
Validation Accuracy for Epoch 4: 0.6367862224578857
Validation Recall for Epoch 4: 0.7428349852561951
Validation Precision for Epoch 4: 0.8168654441833496
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 5 ----------------------------
TRAINING---
Average Training Loss for Epoch 5: 0.08013248122205921
Training Accuracy for Epoch 5: 0.6095048189163208
Training Recall for Epoch 5: 0.8457459211349487
Training Precision for Epoch 5: 0.6857356429100037
VALIDATION---
Average Validation Loss for Epoch 5: 0.3168831765651703
Validation Accuracy for Epoch 5: 0.5949535369873047
Validation Recall for Epoch 5: 0.5949535369873047
Validation Precision for Epoch 5: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 6 ----------------------------
TRAINING---
Average Training Loss for Epoch 6: 0.08072609063417423
Training Accuracy for Epoch 6: 0.6070122718811035
Training Recall for Epoch 6: 0.8119581937789917
Training Precision for Epoch 6: 0.7063031792640686
VALIDATION---
Average Validation Loss for Epoch 6: 0.16799592971801758
Validation Accuracy for Epoch 6: 0.6215139627456665
Validation Recall for Epoch 6: 0.7410926222801208
Validation Precision for Epoch 6: 0.7938931584358215
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 7 ----------------------------
TRAINING---
Average Training Loss for Epoch 7: 0.07847812819338272
Training Accuracy for Epoch 7: 0.6370887160301208
Training Recall for Epoch 7: 0.9050991535186768
Training Precision for Epoch 7: 0.682692289352417
VALIDATION---
Average Validation Loss for Epoch 7: 0.35577377676963806
Validation Accuracy for Epoch 7: 0.6334661245346069
Validation Recall for Epoch 7: 0.7429906725883484
Validation Precision for Epoch 7: 0.8112244606018066
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 8 ----------------------------
TRAINING---
Average Training Loss for Epoch 8: 0.07982099141669297
Training Accuracy for Epoch 8: 0.6256231069564819
Training Recall for Epoch 8: 0.9731196761131287
Training Precision for Epoch 8: 0.6366249322891235
VALIDATION---
Average Validation Loss for Epoch 8: 0.1329115927219391
Validation Accuracy for Epoch 8: 0.652058482170105
Validation Recall for Epoch 8: 0.8598949313163757
Validation Precision for Epoch 8: 0.7295690774917603
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 9 ----------------------------
TRAINING---
Average Training Loss for Epoch 9: 0.07628599779377668
Training Accuracy for Epoch 9: 0.6699900031089783
Training Recall for Epoch 9: 0.9467011094093323
Training Precision for Epoch 9: 0.6962528228759766
VALIDATION---
Average Validation Loss for Epoch 9: 0.18564797937870026
Validation Accuracy for Epoch 9: 0.6401062607765198
Validation Recall for Epoch 9: 0.8339100480079651
Validation Precision for Epoch 9: 0.7336377501487732
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 10 ----------------------------
TRAINING---
Average Training Loss for Epoch 10: 0.07579770313817902
Training Accuracy for Epoch 10: 0.6773014068603516
Training Recall for Epoch 10: 0.9525589942932129
Training Precision for Epoch 10: 0.7009458541870117
VALIDATION---
Average Validation Loss for Epoch 10: 0.16158625483512878
Validation Accuracy for Epoch 10: 0.6633466482162476
Validation Recall for Epoch 10: 0.8763157725334167
Validation Precision for Epoch 10: 0.7318681478500366
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 11 ----------------------------
TRAINING---
Average Training Loss for Epoch 11: 0.07591065623041957
Training Accuracy for Epoch 11: 0.6824526190757751
Training Recall for Epoch 11: 0.9526792168617249
Training Precision for Epoch 11: 0.7063983678817749
VALIDATION---
Average Validation Loss for Epoch 11: 0.1488172560930252
Validation Accuracy for Epoch 11: 0.6646746397018433
Validation Recall for Epoch 11: 0.8742358088493347
Validation Precision for Epoch 11: 0.7349485754966736
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 12 ----------------------------
TRAINING---
Average Training Loss for Epoch 12: 0.07419328631981142
Training Accuracy for Epoch 12: 0.6889331936836243
Training Recall for Epoch 12: 0.9509174227714539
Training Precision for Epoch 12: 0.7143349647521973
VALIDATION---
Average Validation Loss for Epoch 12: 0.17469094693660736
Validation Accuracy for Epoch 12: 0.6640106439590454
Validation Recall for Epoch 12: 0.8680555820465088
Validation Precision for Epoch 12: 0.738552451133728
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 13 ----------------------------
TRAINING---
Average Training Loss for Epoch 13: 0.0733621331366957
Training Accuracy for Epoch 13: 0.6940844058990479
Training Recall for Epoch 13: 0.9527828693389893
Training Precision for Epoch 13: 0.7188091278076172
VALIDATION---
Average Validation Loss for Epoch 13: 0.12770675122737885
Validation Accuracy for Epoch 13: 0.6693227291107178
Validation Recall for Epoch 13: 0.8674699068069458
Validation Precision for Epoch 13: 0.7455621361732483
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 14 ----------------------------
TRAINING---
Average Training Loss for Epoch 14: 0.0725649096772456
Training Accuracy for Epoch 14: 0.7027251720428467
Training Recall for Epoch 14: 0.947356641292572
Training Precision for Epoch 14: 0.7312813401222229
VALIDATION---
Average Validation Loss for Epoch 14: 0.12314706295728683
Validation Accuracy for Epoch 14: 0.6925631165504456
Validation Recall for Epoch 14: 0.9093286991119385
Validation Precision for Epoch 14: 0.7439372539520264
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 15 ----------------------------
TRAINING---
Average Training Loss for Epoch 15: 0.07327859048215049
Training Accuracy for Epoch 15: 0.6965769529342651
Training Recall for Epoch 15: 0.9490604400634766
Training Precision for Epoch 15: 0.7236319780349731
VALIDATION---
Average Validation Loss for Epoch 15: 0.08974526822566986
Validation Accuracy for Epoch 15: 0.6918990612030029
Validation Recall for Epoch 15: 0.9229406714439392
Validation Precision for Epoch 15: 0.7343199253082275
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 16 ----------------------------
TRAINING---
Average Training Loss for Epoch 16: 0.07233288196797939
Training Accuracy for Epoch 16: 0.7087072134017944
Training Recall for Epoch 16: 0.9425414204597473
Training Precision for Epoch 16: 0.740708589553833
VALIDATION---
Average Validation Loss for Epoch 16: 0.084181047976017
Validation Accuracy for Epoch 16: 0.6992031931877136
Validation Recall for Epoch 16: 0.9204545617103577
Validation Precision for Epoch 16: 0.7441695928573608
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 17 ----------------------------
TRAINING---
Average Training Loss for Epoch 17: 0.07189776841396309
Training Accuracy for Epoch 17: 0.7125290632247925
Training Recall for Epoch 17: 0.942417562007904
Training Precision for Epoch 17: 0.7449617981910706
VALIDATION---
Average Validation Loss for Epoch 17: 0.08231183141469955
Validation Accuracy for Epoch 17: 0.701195240020752
Validation Recall for Epoch 17: 0.9214659929275513
Validation Precision for Epoch 17: 0.7457627058029175
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 18 ----------------------------
TRAINING---
Average Training Loss for Epoch 18: 0.07007157138244541
Training Accuracy for Epoch 18: 0.7259886860847473
Training Recall for Epoch 18: 0.9299702048301697
Training Precision for Epoch 18: 0.7679733037948608
VALIDATION---
Average Validation Loss for Epoch 18: 0.07903099060058594
Validation Accuracy for Epoch 18: 0.7144754528999329
Validation Recall for Epoch 18: 0.9180887341499329
Validation Precision for Epoch 18: 0.7631205916404724
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
EPOCH 19 ----------------------------
TRAINING---
Average Training Loss for Epoch 19: 0.06844499469764131
Training Accuracy for Epoch 19: 0.7352941036224365
Training Recall for Epoch 19: 0.92476487159729
Training Precision for Epoch 19: 0.7820784449577332
VALIDATION---
Average Validation Loss for Epoch 19: 0.0758434534072876
Validation Accuracy for Epoch 19: 0.7370518445968628
Validation Recall for Epoch 19: 0.9017059206962585
Validation Precision for Epoch 19: 0.8014440536499023
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/test_2
```