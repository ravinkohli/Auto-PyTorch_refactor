"""
======================
Image Classification
======================
"""
import torchvision.datasets

from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline


# Get the training data for tabular classification
trainset = torchvision.datasets.FashionMNIST(root='../datasets/', train=True, download=True)
X_train = trainset.data.numpy().reshape(-1, trainset.data.shape[1]*trainset.data.shape[2])

numerical_columns = list(range(X_train.shape[1]))
categorical_columns = []
# Create a proof of concept pipeline!
dataset_properties = {
    'categorical_columns': categorical_columns,
    'numerical_columns': numerical_columns
}
pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)

# Configuration space
pipeline_cs = pipeline.get_hyperparameter_search_space()
print("Pipeline CS:\n", '_' * 40, f"\n{pipeline_cs}")
config = pipeline_cs.sample_configuration()
print("Pipeline Random Config:\n", '_' * 40, f"\n{config}")
pipeline.set_hyperparameters(config)

# Fit the pipeline
print("Fitting the pipeline...")

pipeline.fit(X={
    'categorical_columns': categorical_columns,
    'numerical_columns': numerical_columns,
    'num_features': len(numerical_columns),
    'num_classes': 10,
    'train': X_train
})

# Showcase some components of the pipeline
print(pipeline)