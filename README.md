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


Docker:
I've been using:
`docker build . -f Dockerfile -t mlops_teamproject:latest` and `docker run -p 8000:8000 mlops_teamproject:latest`, although the docker run will require attached volumes for data/models