# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and Scikit-learn's Logistic Regression model. First, I created a custom model and optimized its hyperparameters using HyperDrive. Next, we compared the model with the best hyperparameters to an Azure AutoML run.

## Summary
This dataset contains data about the marketing campaigns of a banking institution. Our goal is to predict whether the client will subscribe to the bank or not.

The best performing model was the AutoML model using the Voting Ensemble Algorithm with an accuracy of 0.916. The custom Logistic regression model had an accuracy of 0.912 with 200 iterations and C=0.01.

## Scikit-learn Pipeline
**Pipeline architecture**

![pipeline-architecture](images/pipeline_architecture.png)

**Parameter Sampler**

The hyperparameter sampler was defined in `udacity-project.ipynb` as below:

```python
ps = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,100),
        '--max_iter': choice(50,100,200)
    }
)
```

Here, `C` is the regularization strength, and `max_iter` defines the total number of iterations. Some options available in the Azure sampling library are `RandomParameterSampling`, `GridParameterSampling`, `BayesianParameterSamping`, etc. Out of these, I used `RandomParameterSampling` as it is fast and supports early termination  for low-performance runs.

**Early Stopping Policy**

I used the BanditPolicy for early stopping, which is added to `udacity-project.ipynb` as follows:

```python
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```

Here,`evaluation_interval` is an optional argument that represents the frequency for applying the policy. The `slack_factor` argument defines the amount of slack allowed with respect to the best performing training run.

## AutoML
**AutoML config**

The AutoML parameter config was defined as follows:

```python
automl_config = AutoMLConfig(
    compute_target = compute_target,
    experiment_timeout_minutes=15,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    enable_onnx_compatible_models=True,
    n_cross_validations=2)
```

The above config ensures that we run various classification algorithms such as Voting Ensemble, Gradient Boosting, etc.,  and rank them on the basis of `accuracy`. In the above config, `experiment_timeout_minutes= 15` ensures that the experiment should stop after 15 mins. This is primarily done to reduce costs. The value of `n_cross_validations` is set to 2 to deal with overfitting issues.

## Pipeline comparison

| Model                    | Accuracy |
| ------------------------ | -------- |
| HyperDrive               | 0.912    |
| AutoML (Voting Ensemble) | 0.916    |

Both the models have similar performance. The AutoML model is marginally better because of slightly better accuracy and an AUC value of 0.94. Since Voting Ensemble model has a high AUC value it would be better in distinguishing between the 2 classes in our problem. We could improve the AutoML model further if we let it run for more time. 

## Future work
**Areas of improvement**

- Tackle class imbalance by oversampling of minority class
- Try other custom algorithms
- Run AutoML for a longer time to get better results.

## Cluster clean up
To ensure that I don't incur extra charges, I deleted the compute cluster using the following code in `udacity-project.ipynb`

```python
# Delete cluster
compute_target.delete()
```
