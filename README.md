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

Test 2 model (CNN used for prediction):
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


NN for comparison:
```
{
  "dataset_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/data/initial",
  "test_size": 0.2,
  "batch_size": 8,
  "learning_rate": 0.001,
  "momentum": 0.9,
  "num_epochs": 20,
  "save_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1",
  "model_type": "nn",
  "confusion_matrix_save_path": "/home/user/jhu/mlops/AI_MLOps_Team1_Project/nn_1_cm.png"
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


NN For comparison training:
```
EPOCH 0 ----------------------------
TRAINING---
Average Training Loss for Epoch 0: 6.852734291026583
Training Accuracy for Epoch 0: 0.44915252923965454
Training Recall for Epoch 0: 0.9963140487670898
Training Precision for Epoch 0: 0.4499001204967499
VALIDATION---
Average Validation Loss for Epoch 0: 6.964751720428467
Validation Accuracy for Epoch 0: 0.44355911016464233
Validation Recall for Epoch 0: 1.0
Validation Precision for Epoch 0: 0.44355911016464233
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 1 ----------------------------
TRAINING---
Average Training Loss for Epoch 1: 5.488250427317009
Training Accuracy for Epoch 1: 0.4714190661907196
Training Recall for Epoch 1: 0.8032276034355164
Training Precision for Epoch 1: 0.532970130443573
VALIDATION---
Average Validation Loss for Epoch 1: 0.27882784605026245
Validation Accuracy for Epoch 1: 0.4515272378921509
Validation Recall for Epoch 1: 1.0
Validation Precision for Epoch 1: 0.4515272378921509
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 2 ----------------------------
TRAINING---
Average Training Loss for Epoch 2: 5.600417581496741
Training Accuracy for Epoch 2: 0.5511797666549683
Training Recall for Epoch 2: 0.551638126373291
Training Precision for Epoch 2: 0.9984948635101318
VALIDATION---
Average Validation Loss for Epoch 2: 5.569389343261719
Validation Accuracy for Epoch 2: 0.5564409494400024
Validation Recall for Epoch 2: 0.5564408898353577
Validation Precision for Epoch 2: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 3 ----------------------------
TRAINING---
Average Training Loss for Epoch 3: 5.603748061641897
Training Accuracy for Epoch 3: 0.5516782999038696
Training Recall for Epoch 3: 0.5516782999038696
Training Precision for Epoch 3: 1.0
VALIDATION---
Average Validation Loss for Epoch 3: 5.569389343261719
Validation Accuracy for Epoch 3: 0.5564409494400024
Validation Recall for Epoch 3: 0.5564408898353577
Validation Precision for Epoch 3: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 4 ----------------------------
TRAINING---
Average Training Loss for Epoch 4: 5.6037525685374705
Training Accuracy for Epoch 4: 0.5516782999038696
Training Recall for Epoch 4: 0.5516782999038696
Training Precision for Epoch 4: 1.0
VALIDATION---
Average Validation Loss for Epoch 4: 5.569389343261719
Validation Accuracy for Epoch 4: 0.5564409494400024
Validation Recall for Epoch 4: 0.5564408898353577
Validation Precision for Epoch 4: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 5 ----------------------------
TRAINING---
Average Training Loss for Epoch 5: 5.603757503936757
Training Accuracy for Epoch 5: 0.5516782999038696
Training Recall for Epoch 5: 0.5516782999038696
Training Precision for Epoch 5: 1.0
VALIDATION---
Average Validation Loss for Epoch 5: 5.569389343261719
Validation Accuracy for Epoch 5: 0.5564409494400024
Validation Recall for Epoch 5: 0.5564408898353577
Validation Precision for Epoch 5: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 6 ----------------------------
TRAINING---
Average Training Loss for Epoch 6: 5.6037494194155
Training Accuracy for Epoch 6: 0.5516782999038696
Training Recall for Epoch 6: 0.5516782999038696
Training Precision for Epoch 6: 1.0
VALIDATION---
Average Validation Loss for Epoch 6: 5.569389343261719
Validation Accuracy for Epoch 6: 0.5564409494400024
Validation Recall for Epoch 6: 0.5564408898353577
Validation Precision for Epoch 6: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 7 ----------------------------
TRAINING---
Average Training Loss for Epoch 7: 5.603749724312374
Training Accuracy for Epoch 7: 0.5516782999038696
Training Recall for Epoch 7: 0.5516782999038696
Training Precision for Epoch 7: 1.0
VALIDATION---
Average Validation Loss for Epoch 7: 5.569389343261719
Validation Accuracy for Epoch 7: 0.5564409494400024
Validation Recall for Epoch 7: 0.5564408898353577
Validation Precision for Epoch 7: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 8 ----------------------------
TRAINING---
Average Training Loss for Epoch 8: 5.603745126137903
Training Accuracy for Epoch 8: 0.5516782999038696
Training Recall for Epoch 8: 0.5516782999038696
Training Precision for Epoch 8: 1.0
VALIDATION---
Average Validation Loss for Epoch 8: 5.569389343261719
Validation Accuracy for Epoch 8: 0.5564409494400024
Validation Recall for Epoch 8: 0.5564408898353577
Validation Precision for Epoch 8: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 9 ----------------------------
TRAINING---
Average Training Loss for Epoch 9: 5.603752368231
Training Accuracy for Epoch 9: 0.5516782999038696
Training Recall for Epoch 9: 0.5516782999038696
Training Precision for Epoch 9: 1.0
VALIDATION---
Average Validation Loss for Epoch 9: 5.569389343261719
Validation Accuracy for Epoch 9: 0.5564409494400024
Validation Recall for Epoch 9: 0.5564408898353577
Validation Precision for Epoch 9: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 10 ----------------------------
TRAINING---
Average Training Loss for Epoch 10: 5.603753448998504
Training Accuracy for Epoch 10: 0.5516782999038696
Training Recall for Epoch 10: 0.5516782999038696
Training Precision for Epoch 10: 1.0
VALIDATION---
Average Validation Loss for Epoch 10: 5.569389343261719
Validation Accuracy for Epoch 10: 0.5564409494400024
Validation Recall for Epoch 10: 0.5564408898353577
Validation Precision for Epoch 10: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 11 ----------------------------
TRAINING---
Average Training Loss for Epoch 11: 5.603747565946772
Training Accuracy for Epoch 11: 0.5516782999038696
Training Recall for Epoch 11: 0.5516782999038696
Training Precision for Epoch 11: 1.0
VALIDATION---
Average Validation Loss for Epoch 11: 5.569389343261719
Validation Accuracy for Epoch 11: 0.5564409494400024
Validation Recall for Epoch 11: 0.5564408898353577
Validation Precision for Epoch 11: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 12 ----------------------------
TRAINING---
Average Training Loss for Epoch 12: 5.603754804236583
Training Accuracy for Epoch 12: 0.5516782999038696
Training Recall for Epoch 12: 0.5516782999038696
Training Precision for Epoch 12: 1.0
VALIDATION---
Average Validation Loss for Epoch 12: 5.569389343261719
Validation Accuracy for Epoch 12: 0.5564409494400024
Validation Recall for Epoch 12: 0.5564408898353577
Validation Precision for Epoch 12: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 13 ----------------------------
TRAINING---
Average Training Loss for Epoch 13: 5.603748054035322
Training Accuracy for Epoch 13: 0.5516782999038696
Training Recall for Epoch 13: 0.5516782999038696
Training Precision for Epoch 13: 1.0
VALIDATION---
Average Validation Loss for Epoch 13: 5.569389343261719
Validation Accuracy for Epoch 13: 0.5564409494400024
Validation Recall for Epoch 13: 0.5564408898353577
Validation Precision for Epoch 13: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 14 ----------------------------
TRAINING---
Average Training Loss for Epoch 14: 5.603755228937009
Training Accuracy for Epoch 14: 0.5516782999038696
Training Recall for Epoch 14: 0.5516782999038696
Training Precision for Epoch 14: 1.0
VALIDATION---
Average Validation Loss for Epoch 14: 5.569389343261719
Validation Accuracy for Epoch 14: 0.5564409494400024
Validation Recall for Epoch 14: 0.5564408898353577
Validation Precision for Epoch 14: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 15 ----------------------------
TRAINING---
Average Training Loss for Epoch 15: 5.603748529446248
Training Accuracy for Epoch 15: 0.5516782999038696
Training Recall for Epoch 15: 0.5516782999038696
Training Precision for Epoch 15: 1.0
VALIDATION---
Average Validation Loss for Epoch 15: 5.569389343261719
Validation Accuracy for Epoch 15: 0.5564409494400024
Validation Recall for Epoch 15: 0.5564408898353577
Validation Precision for Epoch 15: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 16 ----------------------------
TRAINING---
Average Training Loss for Epoch 16: 5.603745822139498
Training Accuracy for Epoch 16: 0.5516782999038696
Training Recall for Epoch 16: 0.5516782999038696
Training Precision for Epoch 16: 1.0
VALIDATION---
Average Validation Loss for Epoch 16: 5.569389343261719
Validation Accuracy for Epoch 16: 0.5564409494400024
Validation Recall for Epoch 16: 0.5564408898353577
Validation Precision for Epoch 16: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 17 ----------------------------
TRAINING---
Average Training Loss for Epoch 17: 5.603749802913646
Training Accuracy for Epoch 17: 0.5516782999038696
Training Recall for Epoch 17: 0.5516782999038696
Training Precision for Epoch 17: 1.0
VALIDATION---
Average Validation Loss for Epoch 17: 5.569389343261719
Validation Accuracy for Epoch 17: 0.5564409494400024
Validation Recall for Epoch 17: 0.5564408898353577
Validation Precision for Epoch 17: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 18 ----------------------------
TRAINING---
Average Training Loss for Epoch 18: 5.603751677934337
Training Accuracy for Epoch 18: 0.5516782999038696
Training Recall for Epoch 18: 0.5516782999038696
Training Precision for Epoch 18: 1.0
VALIDATION---
Average Validation Loss for Epoch 18: 5.569389343261719
Validation Accuracy for Epoch 18: 0.5564409494400024
Validation Recall for Epoch 18: 0.5564408898353577
Validation Precision for Epoch 18: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
EPOCH 19 ----------------------------
TRAINING---
Average Training Loss for Epoch 19: 5.6037495753502835
Training Accuracy for Epoch 19: 0.5516782999038696
Training Recall for Epoch 19: 0.5516782999038696
Training Precision for Epoch 19: 1.0
VALIDATION---
Average Validation Loss for Epoch 19: 5.569389343261719
Validation Accuracy for Epoch 19: 0.5564409494400024
Validation Recall for Epoch 19: 0.5564408898353577
Validation Precision for Epoch 19: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/nn_1
```


CNN TEST 1
```
EPOCH 0 ----------------------------
TRAINING---
Average Training Loss for Epoch 0: 0.08390040278939878
Training Accuracy for Epoch 0: 0.5558325052261353
Training Recall for Epoch 0: 0.7100403308868408
Training Precision for Epoch 0: 0.7190455794334412
VALIDATION---
Average Validation Loss for Epoch 0: 0.15676194429397583
Validation Accuracy for Epoch 0: 0.5630810260772705
Validation Recall for Epoch 0: 0.5630810260772705
Validation Precision for Epoch 0: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 1 ----------------------------
TRAINING---
Average Training Loss for Epoch 1: 0.0777516485816787
Training Accuracy for Epoch 1: 0.6164838671684265
Training Recall for Epoch 1: 0.7293099761009216
Training Precision for Epoch 1: 0.7993966937065125
VALIDATION---
Average Validation Loss for Epoch 1: 0.07535649091005325
Validation Accuracy for Epoch 1: 0.6580345630645752
Validation Recall for Epoch 1: 0.7078571319580078
Validation Precision for Epoch 1: 0.9033728241920471
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 2 ----------------------------
TRAINING---
Average Training Loss for Epoch 2: 0.07519889137363782
Training Accuracy for Epoch 2: 0.6354270577430725
Training Recall for Epoch 2: 0.7063169479370117
Training Precision for Epoch 2: 0.8635953068733215
VALIDATION---
Average Validation Loss for Epoch 2: 0.08266562223434448
Validation Accuracy for Epoch 2: 0.5717131495475769
Validation Recall for Epoch 2: 0.5736175775527954
Validation Precision for Epoch 2: 0.994226336479187
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 3 ----------------------------
TRAINING---
Average Training Loss for Epoch 3: 0.07231166689595082
Training Accuracy for Epoch 3: 0.6523761749267578
Training Recall for Epoch 3: 0.7458206415176392
Training Precision for Epoch 3: 0.8388888835906982
VALIDATION---
Average Validation Loss for Epoch 3: 0.08627156168222427
Validation Accuracy for Epoch 3: 0.6673306822776794
Validation Recall for Epoch 3: 0.7137784361839294
Validation Precision for Epoch 3: 0.9111514091491699
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 4 ----------------------------
TRAINING---
Average Training Loss for Epoch 4: 0.06739239262472052
Training Accuracy for Epoch 4: 0.7007311582565308
Training Recall for Epoch 4: 0.7387876510620117
Training Precision for Epoch 4: 0.9315219521522522
VALIDATION---
Average Validation Loss for Epoch 4: 0.534321665763855
Validation Accuracy for Epoch 4: 0.7403718829154968
Validation Recall for Epoch 4: 0.857692301273346
Validation Precision for Epoch 4: 0.844057559967041
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 5 ----------------------------
TRAINING---
Average Training Loss for Epoch 5: 6.867923984538681
Training Accuracy for Epoch 5: 0.4504818916320801
Training Recall for Epoch 5: 0.999262809753418
Training Precision for Epoch 5: 0.45063164830207825
VALIDATION---
Average Validation Loss for Epoch 5: 7.0662922859191895
Validation Accuracy for Epoch 5: 0.4369190037250519
Validation Recall for Epoch 5: 1.0
Validation Precision for Epoch 5: 0.4369190037250519
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 6 ----------------------------
TRAINING---
Average Training Loss for Epoch 6: 6.871438422287053
Training Accuracy for Epoch 6: 0.4503157138824463
Training Recall for Epoch 6: 1.0
Training Precision for Epoch 6: 0.4503157138824463
VALIDATION---
Average Validation Loss for Epoch 6: 7.013631820678711
Validation Accuracy for Epoch 6: 0.4369190037250519
Validation Recall for Epoch 6: 1.0
Validation Precision for Epoch 6: 0.4369190037250519
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 7 ----------------------------
TRAINING---
Average Training Loss for Epoch 7: 3.534325815150412
Training Accuracy for Epoch 7: 0.49883681535720825
Training Recall for Epoch 7: 0.6986269354820251
Training Precision for Epoch 7: 0.635612964630127
VALIDATION---
Average Validation Loss for Epoch 7: 0.14670202136039734
Validation Accuracy for Epoch 7: 0.509296178817749
Validation Recall for Epoch 7: 0.6985428333282471
Validation Precision for Epoch 7: 0.6527659296989441
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 8 ----------------------------
TRAINING---
Average Training Loss for Epoch 8: 0.08311438101417165
Training Accuracy for Epoch 8: 0.5546693205833435
Training Recall for Epoch 8: 0.5657626986503601
Training Precision for Epoch 8: 0.9658564925193787
VALIDATION---
Average Validation Loss for Epoch 8: 0.08046762645244598
Validation Accuracy for Epoch 8: 0.5630810260772705
Validation Recall for Epoch 8: 0.5630810260772705
Validation Precision for Epoch 8: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 9 ----------------------------
TRAINING---
Average Training Loss for Epoch 9: 0.08147039490136922
Training Accuracy for Epoch 9: 0.5729478001594543
Training Recall for Epoch 9: 0.5955094695091248
Training Precision for Epoch 9: 0.937976062297821
VALIDATION---
Average Validation Loss for Epoch 9: 0.10784531384706497
Validation Accuracy for Epoch 9: 0.6241700053215027
Validation Recall for Epoch 9: 0.7292474508285522
Validation Precision for Epoch 9: 0.8124459981918335
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 10 ----------------------------
TRAINING---
Average Training Loss for Epoch 10: 0.08057058325496057
Training Accuracy for Epoch 10: 0.5958790183067322
Training Recall for Epoch 10: 0.648697555065155
Training Precision for Epoch 10: 0.879784107208252
VALIDATION---
Average Validation Loss for Epoch 10: 0.0776902437210083
Validation Accuracy for Epoch 10: 0.5630810260772705
Validation Recall for Epoch 10: 0.5630810260772705
Validation Precision for Epoch 10: 1.0
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 11 ----------------------------
TRAINING---
Average Training Loss for Epoch 11: 0.07487328370600374
Training Accuracy for Epoch 11: 0.6935858726501465
Training Recall for Epoch 11: 0.8652570247650146
Training Precision for Epoch 11: 0.7775707840919495
VALIDATION---
Average Validation Loss for Epoch 11: 0.08842067420482635
Validation Accuracy for Epoch 11: 0.7423639297485352
Validation Recall for Epoch 11: 0.8929712176322937
Validation Precision for Epoch 11: 0.8148688077926636
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 12 ----------------------------
TRAINING---
Average Training Loss for Epoch 12: 0.06881767304613567
Training Accuracy for Epoch 12: 0.73762047290802
Training Recall for Epoch 12: 0.8945989608764648
Training Precision for Epoch 12: 0.8078252673149109
VALIDATION---
Average Validation Loss for Epoch 12: 0.11224338412284851
Validation Accuracy for Epoch 12: 0.7543160915374756
Validation Recall for Epoch 12: 0.9059011340141296
Validation Precision for Epoch 12: 0.818443775177002
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 13 ----------------------------
TRAINING---
Average Training Loss for Epoch 13: 0.0713952596278613
Training Accuracy for Epoch 13: 0.71003657579422
Training Recall for Epoch 13: 0.867614209651947
Training Precision for Epoch 13: 0.7963101267814636
VALIDATION---
Average Validation Loss for Epoch 13: 0.1114238053560257
Validation Accuracy for Epoch 13: 0.7490040063858032
Validation Recall for Epoch 13: 0.9126213788986206
Validation Precision for Epoch 13: 0.8068669438362122
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 14 ----------------------------
TRAINING---
Average Training Loss for Epoch 14: 0.08263917983162637
Training Accuracy for Epoch 14: 0.5442007184028625
Training Recall for Epoch 14: 0.6348129510879517
Training Precision for Epoch 14: 0.7922109365463257
VALIDATION---
Average Validation Loss for Epoch 14: 0.08854599297046661
Validation Accuracy for Epoch 14: 0.5391766428947449
Validation Recall for Epoch 14: 0.557692289352417
Validation Precision for Epoch 14: 0.94199538230896
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 15 ----------------------------
TRAINING---
Average Training Loss for Epoch 15: 0.07173844451631332
Training Accuracy for Epoch 15: 0.7025589942932129
Training Recall for Epoch 15: 0.8874894976615906
Training Precision for Epoch 15: 0.771251380443573
VALIDATION---
Average Validation Loss for Epoch 15: 0.21781693398952484
Validation Accuracy for Epoch 15: 0.7529881000518799
Validation Recall for Epoch 15: 0.8603945374488831
Validation Precision for Epoch 15: 0.8577912449836731
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 16 ----------------------------
TRAINING---
Average Training Loss for Epoch 16: 0.08094436490840791
Training Accuracy for Epoch 16: 0.604519784450531
Training Recall for Epoch 16: 0.7747018933296204
Training Precision for Epoch 16: 0.7334677577018738
VALIDATION---
Average Validation Loss for Epoch 16: 0.08436135202646255
Validation Accuracy for Epoch 16: 0.7231075763702393
Validation Recall for Epoch 16: 0.9014900922775269
Validation Precision for Epoch 16: 0.7851477861404419
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 17 ----------------------------
TRAINING---
Average Training Loss for Epoch 17: 0.06772631683859488
Training Accuracy for Epoch 17: 0.7306414246559143
Training Recall for Epoch 17: 0.9148980379104614
Training Precision for Epoch 17: 0.7839186787605286
VALIDATION---
Average Validation Loss for Epoch 17: 0.08457807451486588
Validation Accuracy for Epoch 17: 0.733731746673584
Validation Recall for Epoch 17: 0.8840000033378601
Validation Precision for Epoch 17: 0.8119029998779297
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 18 ----------------------------
TRAINING---
Average Training Loss for Epoch 18: 0.06452855238014378
Training Accuracy for Epoch 18: 0.7524093985557556
Training Recall for Epoch 18: 0.9014533162117004
Training Precision for Epoch 18: 0.8198443055152893
VALIDATION---
Average Validation Loss for Epoch 18: 0.08306248486042023
Validation Accuracy for Epoch 18: 0.7622842192649841
Validation Recall for Epoch 18: 0.8989819884300232
Validation Precision for Epoch 18: 0.8336964249610901
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
EPOCH 19 ----------------------------
TRAINING---
Average Training Loss for Epoch 19: 0.07744521122016
Training Accuracy for Epoch 19: 0.6025257706642151
Training Recall for Epoch 19: 0.9112842679023743
Training Precision for Epoch 19: 0.6400706171989441
VALIDATION---
Average Validation Loss for Epoch 19: 0.08669806271791458
Validation Accuracy for Epoch 19: 0.6766268610954285
Validation Recall for Epoch 19: 0.9400368928909302
Validation Precision for Epoch 19: 0.7071478366851807
Saving to: /home/user/jhu/mlops/AI_MLOps_Team1_Project/models/cnn_test1
```