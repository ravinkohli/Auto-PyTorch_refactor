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

from autoPyTorch.constants import TABULAR_CLASSIFICATION, TASK_TYPES_TO_STRING
from autoPyTorch.data.xy_data_manager import XYDataManager
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.utils.backend import create
from autoPyTorch.utils.stopwatch import StopWatch
from autoPyTorch.evaluation.train_evaluator import eval_holdout
from autoPyTorch.utils.pipeline import get_configuration_space
from autoPyTorch.utils.logging_ import setup_logger, start_log_server


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


def get_data_to_train() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, typing.Optional[np.ndarray]]:
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
    # feat_type = np.random.choice(['numerical', 'categorical'], (X_train.shape[1]))
    return X_train, X_test, y_train, y_test, None


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
        task=TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
        dataset_name='Australian',
        feat_type=feat_type,
    )
    backend.save_datamanager(datamanager)

    # backend.save_fit_dictionary(fit_dictionary)

    pipeline_cs = get_configuration_space(datamanager.info)

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
    run_history, trajectory, budget = smbo.run_smbo(eval_holdout)

    for k, v in run_history.data.items():
        print(f"{k}->{v.additional_info}")
