# AI_MLOps_Team1_Project

To run, I've been using: `uvicorn main:app --reload`

An example input to the training endpoint:
```
{
  "dataset_path": "/home/user/jhu/mlops/teamproject/data/initial",
  "test_size": 0.2,
  "batch_size": 32,
  "num_epochs": 10,
  "save_path": "/home/user/jhu/mlops/teamproject/models/test_1",
  "model_type": "cnn"
}
```