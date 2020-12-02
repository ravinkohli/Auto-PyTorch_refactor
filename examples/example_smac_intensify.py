import typing
import os

import dask
import dask.distributed

import numpy as np

import sklearn.datasets
import sklearn.model_selection
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score

from smac.tae import StatusType

from autoPyTorch.constants import TABULAR_CLASSIFICATION, TASK_TYPES_TO_STRING
from autoPyTorch.data.xy_data_manager import XYDataManager
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.utils.backend import create
from autoPyTorch.utils.stopwatch import StopWatch
from autoPyTorch.utils.pipeline import get_configuration_space


def get_data_to_train() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    feat_type = np.random.choice(['numerical', 'categorical'], (X_train.shape[1]))
    return X_train, X_test, y_train, y_test, feat_type


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
    X_train = check_array(fit_dictionary[0])
    y_train = check_array(fit_dictionary[2], ensure_2d=False)
    X_test = check_array(fit_dictionary[1])
    y_test = check_array(fit_dictionary[3], ensure_2d=False)
    feat_type = fit_dictionary[4]
    datamanager = XYDataManager(
        X=X_train, y=y_train,
        X_test=X_test,
        y_test=y_test,
        task=TABULAR_CLASSIFICATION,
        dataset_name='Australian',
        feat_type=feat_type,
    )
    backend.save_datamanager(datamanager)

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
        config_space=get_configuration_space(datamanager.info),
        dataset_name='Australian',
        backend=backend,
        total_walltime_limit=100,
        dask_client=dask_client,
        func_eval_time_limit=30,
        memory_limit=1024,
        metric=get_metrics(dataset_properties=dict({'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
                                                    'output_type': datamanager.info['output_type']})),
        watcher=StopWatch(),
        n_jobs=2,
        ensemble_callback=None,
    )

    # Then run the optimization
    run_history, trajectory, budget = smbo.run_smbo()

    for k, v in run_history.data.items():
        print(f"{k}->{v}")
