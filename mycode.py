from azureml.train.automl import AutoMLConfig
from train import clean_data
from sklearn.model_selection import train_test_split
import pandas as pd

, y = clean_data(ds)
x_train, x_test, y_train, y_test = train_test_split(x, y)

df = pd.concat([x, y], axis=1)
train_data, test_data = train_test_split(
    df, test_size=0.1, random_state=42)

training = pd.concat([x_train, y_train], axis=1)
test = pd.concat([x_test, y_test], axis=1)

if not os.path.isdir('data'):
    os.mkdir('data')

pd.DataFrame(training).to_csv('data/train.csv', index=False)
pd.DataFrame(test).to_csv('data/test.csv', index=False)

ds_def = ws.get_default_datastore()
ds_def.upload(src_dir='./data', target_path='bankmarketing',
              overwrite=True, show_progress=True)

training_ds = TabularDatasetFactory.from_delimited_files(
    path=ds_def.path('bankmarketing/train.csv'))
test_ds = TabularDatasetFactory.from_delimited_files(
    path=ds_def.path('bankmarketing/test.csv'))

# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=training_ds,
    label_column_name='y',
    n_cross_validations=2,
    compute_target=cpu_cluster)


####################################
#
best_run = run.get_best_child()
print(fitted_model.steps)

model_name = best_run.properties['model_name']
description = 'AutoML forecast example'
tags = None

model = run.register_model(model_name=model_name,
                           description=description,
                           tags=tags)
