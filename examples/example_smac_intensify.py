# import typing
# import os
#
# import dask
# import dask.distributed
#
# import numpy as np
#
# import sklearn.datasets
# import sklearn.model_selection
# from sklearn.utils import check_array
# from sklearn.utils.multiclass import type_of_target
# from sklearn.metrics import accuracy_score
#
# from smac.tae import StatusType
#
# from autoPyTorch.constants import TABULAR_CLASSIFICATION, TASK_TYPES_TO_STRING
# from autoPyTorch.data.xy_data_manager import XYDataManager
# from autoPyTorch.optimizer.smbo import AutoMLSMBO
# from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
# from autoPyTorch.utils.backend import create
# from autoPyTorch.utils.stopwatch import StopWatch
# from autoPyTorch.utils.pipeline import get_configuration_space
#
#
# def get_data_to_train() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, typing.Optional[np.ndarray]]:
#     """
#     This function returns a fit dictionary that within itself, contains all
#     the information to fit a pipeline
#     """
#
#     # Get the training data for tabular classification
#     # Move to Australian to showcase numerical vs categorical
#     X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
#         X,
#         y,
#         random_state=1,
#         test_size=0.2,
#     )
#     # feat_type = np.random.choice(['numerical', 'categorical'], (X_train.shape[1]))
#     return X_train, X_test, y_train, y_test, None
#
#
# if __name__ == "__main__":
#     # Get data to train
#     fit_dictionary = get_data_to_train()
#     # Build a repository with random fitted models
#     backend = create(temporary_directory='/tmp/autoPyTorch_ensemble_test_tmp',
#                      output_directory='/tmp/autoPyTorch_ensemble_test_out',
#                      delete_tmp_folder_after_terminate=False)
#
#     # Create the directory structure
#     backend._make_internals_directory()
#
#     # Ensemble selection will evaluate performance on the OOF predictions. Store the OOF
#     # Ground truth
#     X_train = check_array(fit_dictionary[0])
#     y_train = check_array(fit_dictionary[2], ensure_2d=False)
#     X_test = check_array(fit_dictionary[1])
#     y_test = check_array(fit_dictionary[3], ensure_2d=False)
#     feat_type = fit_dictionary[4]
#     datamanager = XYDataManager(
#         X=X_train, y=y_train,
#         X_test=X_test,
#         y_test=y_test,
#         task=TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
#         dataset_name='Australian',
#         feat_type=feat_type,
#     )
#     backend.save_datamanager(datamanager)
#
#     # Build a ensemble from the above components
#     # Use dak client here to make sure this is proper working,
#     # as with smac we will have to use a client
#     dask.config.set({'distributed.worker.daemon': False})
#     dask_client = dask.distributed.Client(
#         dask.distributed.LocalCluster(
#             n_workers=2,
#             processes=False,
#             threads_per_worker=1,
#             # We use the temporal directory to save the
#             # dask workers, because deleting workers
#             # more time than deleting backend directories
#             # This prevent an error saying that the worker
#             # file was deleted, so the client could not close
#             # the worker properly
#             local_directory=backend.temporary_directory,
#         )
#     )
#
#     # Make the optimizer
#     smbo = AutoMLSMBO(
#         config_space=get_configuration_space(datamanager.info),
#         dataset_name='Australian',
#         backend=backend,
#         total_walltime_limit=60,
#         dask_client=dask_client,
#         func_eval_time_limit=20,
#         memory_limit=4096,
#         metric=get_metrics(dataset_properties=dict({'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
#                                                     'output_type': datamanager.info['output_type']})),
#         watcher=StopWatch(),
#         n_jobs=2,
#         ensemble_callback=None,
#     )
#
#     # Then run the optimization
#     run_history, trajectory, budget = smbo.run_smbo()
#
#     for k, v in run_history.data.items():
#         print(f"{k}->{v}")
import typing

import dask
import dask.distributed

import numpy as np

import sklearn.datasets
import sklearn.model_selection
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score

from smac.tae import StatusType

from autoPyTorch.constants import TABULAR_CLASSIFICATION
from autoPyTorch.data.xy_data_manager import XYDataManager
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.utils.backend import create
from autoPyTorch.utils.stopwatch import StopWatch


def fit_a_pipeline(
    queue,
    config,
    backend,
    metric,
    seed,
    num_run,
    output_y_hat_optimization,
    include,
    exclude,
    disable_file_output,
    instance,
    init_params,
    budget,
    budget_type,
    resampling_strategy,
    resampling_strategy_args,
) -> None:
    """
    Fits a pipeline with the given constrains
    """

    # Ensemble selection will evaluate performance on the OOF predictions. Store the OOF
    # Ground truth
    fit_dictionary = backend.load_fit_dictionary()
    X_train = check_array(fit_dictionary['X_train'])
    y_train = check_array(fit_dictionary['y_train'], ensure_2d=False)
    X_test = check_array(fit_dictionary['X_test'])
    y_test = check_array(fit_dictionary['y_test'], ensure_2d=False)
    targets = np.take(y_train, fit_dictionary['val_indices'], axis=0)
    backend.save_targets_ensemble(targets)

    pipeline = TabularClassificationPipeline(
        dataset_properties=fit_dictionary['dataset_properties'])

    # Set the provided config by smac
    pipeline.set_hyperparameters(config)

    # Fit the sample configuration
    pipeline.fit(fit_dictionary)

    # Predict using the fit model
    ensemble_predictions = pipeline.predict(
        np.take(X_train, fit_dictionary['val_indices'], axis=0))
    test_predictions = pipeline.predict(X_test)

    backend.save_numrun_to_dir(
        seed=fit_dictionary['seed'],
        idx=num_run,
        budget=fit_dictionary['epochs'],
        model=pipeline,
        cv_model=None,
        ensemble_predictions=ensemble_predictions,
        valid_predictions=None,
        test_predictions=test_predictions,
    )
    train_score = accuracy_score(np.take(y_train, fit_dictionary['val_indices'], axis=0),
                                 np.argmax(ensemble_predictions, axis=1))
    test_score = accuracy_score(y_test, np.argmax(test_predictions, axis=1))
    print(f"Fitted a pipeline {config} with score = {train_score}{test_score}")
    rval_dict = {
        'loss': 1 - train_score,
        'additional_run_info': {'test_loss': 1 - test_score},
        'status': StatusType.SUCCESS}
    queue.put(rval_dict)

    return


def get_data_to_train() -> typing.Tuple[typing.Dict[str, typing.Any]]:
    """
    This function returns a fit dictionary that within itself, contains all
    the information to fit a pipeline
    """

    # Get the training data for tabular classification
    # Move to Australian to showcase numerical vs categorical
    X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1,
        test_size=0.2,
    )

    train_indices, val_indices = sklearn.model_selection.train_test_split(
        list(range(X_train.shape[0])),
        random_state=1,
        test_size=0.25,
    )

    output_type = type_of_target(y)

    # Mock the categories
    categorical_columns = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
    numerical_columns = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    categories = [np.unique(X[a]).tolist() for a in categorical_columns]

    # Create a proof of concept pipeline!
    dataset_properties = {
        'task_type': 'tabular_classification',
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'output_type': output_type,
    }
    # Fit the pipeline
    fit_dictionary = {
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'num_features': X.shape[1],
        'num_classes': len(np.unique(y)),
        'is_small_preprocess': True,
        'categories': categories,
        'X_train': X_train,
        'y_train': y_train,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'X_test': X_test,
        'y_test': y_test,
        'dataset_properties': dataset_properties,
        # Training configuration
        'job_id': 'example_tabular_classification_1',
        'working_dir': '/tmp/example_tabular_classification_1',  # Hopefully generated by backend
        'device': 'cpu',
        'runtime': 100,
        'torch_num_threads': 1,
        'early_stopping': 20,
        'use_tensorboard_logger': True,
        'use_pynisher': False,
        'memory_limit': 2048,
        'metrics_during_training': True,
        'seed': 0,
        'budget_type': 'epochs',
        'epochs': 10.0,
        'id': 0,
    }

    return fit_dictionary


if __name__ == "__main__":

    # Get data to train
    fit_dictionary = get_data_to_train()
    # Build a repository with random fitted models
    backend = create(temporary_directory='/tmp/autoPyTorch_ensemble_test_tmp',
                     output_directory='/tmp/autoPyTorch_ensemble_test_out',
                     delete_tmp_folder_after_terminate=False)

    # Create the directory structure
    backend._make_internals_directory()

    # Ensemble selection will evaluate performance on the OOF predictions. Store the OOF
    # Ground truth
    X_train = check_array(fit_dictionary['X_train'])
    y_train = check_array(fit_dictionary['y_train'], ensure_2d=False)
    X_test = check_array(fit_dictionary['X_test'])
    y_test = check_array(fit_dictionary['y_test'], ensure_2d=False)
    targets = np.take(y_train, fit_dictionary['val_indices'], axis=0)
    datamanager = XYDataManager(
        X_train, y_train,
        X_test=X_test,
        y_test=y_test,
        task=TABULAR_CLASSIFICATION,
        dataset_name=fit_dictionary['job_id'],
        feat_type=None,
    )
    backend.save_datamanager(datamanager)
    backend.save_targets_ensemble(targets)

    backend.save_fit_dictionary(fit_dictionary)

    pipeline = TabularClassificationPipeline(
        dataset_properties=fit_dictionary['dataset_properties'])

    # Configuration space
    pipeline_cs = pipeline.get_hyperparameter_search_space()

    # Build a ensemble from the above components
    # Use dak client here to make sure this is proper working,
    # as with smac we will have to use a client
    dask.config.set({'distributed.worker.daemon': False})
    dask_client = dask.distributed.Client(
        dask.distributed.LocalCluster(
            n_workers=2,
            processes=False,
            threads_per_worker=1,
            # We use the temporal directory to save the
            # dask workers, because deleting workers
            # more time than deleting backend directories
            # This prevent an error saying that the worker
            # file was deleted, so the client could not close
            # the worker properly
            local_directory=backend.temporary_directory,
        )
    )

    # Make the optimizer
    smbo = AutoMLSMBO(
        config_space=pipeline_cs,
        dataset_name='Australian',
        backend=backend,
        total_walltime_limit=100,
        dask_client=dask_client,
        func_eval_time_limit=30,
        memory_limit=1024,
        metric=accuracy,
        watcher=StopWatch(),
        n_jobs=2,
        ensemble_callback=None,
    )

    # Then run the optimization
    run_history, trajectory, budget = smbo.run_smbo(fit_a_pipeline)

    for k, v in run_history.data.items():
        print(f"{k}->{v}")
