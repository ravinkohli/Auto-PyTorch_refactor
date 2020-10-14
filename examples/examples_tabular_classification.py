"""
======================
Tabular Classification
======================
"""
import numpy as np

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


# Get the training data for tabular classification
# https://www.openml.org/d/12
X, y = sklearn.datasets.fetch_openml(data_id=12, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1
)

numerical = X.columns.to_list()
categorical = []
# numerical.remove('att214')

# Create a proof of concept pipeline!
dataset_properties = {
    'categorical_columns': categorical,
    'numerical_columns': numerical
}
pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)

# Configuration space
pipeline_cs = pipeline.get_hyperparameter_search_space()
print("Pipeline CS:\n", '_' * 40, f"\n{pipeline_cs}")
config = pipeline_cs.sample_configuration()
print("Pipeline Random Config:\n", '_' * 40, f"\n{config}")
pipeline.set_hyperparameters(config)

# Fit the pipeline
print("Fitting the pipeline...")
pipeline.fit(X={
    'categorical_columns': categorical,
    'numerical_columns': numerical,
    'num_features': len(X),
    'num_classes': len(np.unique(y)),
    'is_small_preprocess': True,
    'X_train': X_train,
})

# Showcase some components of the pipeline
print(pipeline)
