# Pointcloud approach

We attempt to solve the challenge with pointcloud semantic segmentation.

## Data

Modeling the stacked images as 3D space, we generate pointclouds by sampling while keeping the z-axis information. The pointclouds are saved as `.pcd` files.

## Setup

Using `torch-points3d` is complicated. To fullfill all dependencies, we build a docker image and train our model containerized.

The image can be found on docker-hub: `3llobo/torch3dfinal:latest`

## Training

Once inside the container, we can train the model with the following command:

```bash
python3 train_model.py
```

The test set is processed with:
  
```bash
python3 test_model.py
```

## Results

During training, the model converged and achieved:

| Metric | Training | Validation |
| ------ | -------- | ---------- |
| Acc    | 0.91     | 0.85       |
| F1     | 0.61     | 0.55       |

The validation set is unseen, yet part of the same dataset. The test set was unseen and from a different dataset. The test-set result is a poor F0.5 score of `0.13`.

## TODO

- [x] pcd for all samples
- [x] ml model:
  - [x] What is the input?
  - [x] what is the target?
- [x] convert input and target
- [x] data loader: sqite?
- [x] train model

## Model inprovement
- Accuracy function
- Plot predictions function
- Train on all 3 train pieces
- Use bigger pointclouds!!!
## Eval
- eval model
  - Make eval dataset
  - eval model


- Compare to Kaggle
- check rest of versuvius data.

