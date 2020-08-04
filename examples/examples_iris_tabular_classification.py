"""
======================
Tabular Classification
======================
"""
import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


# Get the training data for tabular classification
X, y = sklearn.datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1
)

# Create a proof of concept pipeline!
pipeline = TabularClassificationPipeline()

# Showcase some components of the pipeline
print("Pipeline contains:\n", '_' * 40)
for i, (stage_name, component) in enumerate(pipeline.named_steps.items()):
    print(f"\tStep {i}: {stage_name}")

pipeline_cs = pipeline.get_hyperparameter_search_space()
print("Pipeline CS:\n", '_' * 40, f"\n{pipeline_cs}")
print("Pipeline Random Config:\n", '_' * 40, f"\n{pipeline_cs.sample_configuration()}")

# TODO: This can only be enable when we sync up on how
# to communicate the steps
print("Fitting the pipeline...")
# pipeline.fit(X_train, y_train)
