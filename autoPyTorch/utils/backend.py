import glob
import os
import pickle
import shutil
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union

import lockfile

import numpy as np

from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.utils import logging_ as logging


__all__ = [
    'Backend'
]


def create(
    temporary_directory: str,
    output_directory: str,
    delete_tmp_folder_after_terminate: bool = True,
    delete_output_folder_after_terminate: bool = True,
) -> 'Backend':
    """
    Creates a backend object that manages disk related transactions

    Args:
        temporary_directory (str): where all temporal data is to be dumped
        output_directory (str): where all predictions are to be output
        delete_tmp_folder_after_terminate (bool): whether to delete the
            temporal directory when then run completes
        delete_output_folder_after_terminate (bool): whether to delete
            the output directory when the run completes

    Returns:
        Backend object
    """
    context = BackendContext(temporary_directory, output_directory,
                             delete_tmp_folder_after_terminate,
                             delete_output_folder_after_terminate,
                             )
    backend = Backend(context)

    return backend


def get_randomized_directory_names(
    temporary_directory: Optional[str] = None,
    output_directory: Optional[str] = None,
) -> Tuple[str, str]:
    """
    If the user does not provide a temporal/output directory,
    one is created automatically with uuid to prevent
    several runs colliding

    Args:
        temporary_directory (str): user provided temporal directory
        output_directory (str): user provided output directory

    Returns:
        temporary_directory automatically generated if needed
        output_directory automatically generated if needed
    """
    uuid_str = str(uuid.uuid1(clock_seq=os.getpid()))

    temporary_directory = (
        temporary_directory
        if temporary_directory
        else os.path.join(
            tempfile.gettempdir(),
            "autoPyTorch_tmp_{}".format(
                uuid_str,
            ),
        )
    )

    output_directory = (
        output_directory
        if output_directory
        else os.path.join(
            tempfile.gettempdir(),
            "autoPyTorch_output_{}".format(
                uuid_str,
            ),
        )
    )

    return temporary_directory, output_directory


class BackendContext(object):

    def __init__(self,
                 temporary_directory: str,
                 output_directory: str,
                 delete_tmp_folder_after_terminate: bool,
                 delete_output_folder_after_terminate: bool,
                 ):

        # Check that the names of tmp_dir and output_dir is not the same.
        if temporary_directory == output_directory and temporary_directory is not None:
            raise ValueError("The temporary and the output directory "
                             "must be different.")

        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate
        # attributes to check that directories were created by autoPyTorch
        self._tmp_dir_created = False
        self._output_dir_created = False

        self._temporary_directory, self._output_directory = (
            get_randomized_directory_names(
                temporary_directory=temporary_directory,
                output_directory=output_directory,
            )
        )
        self._logger = logging.get_logger(__name__)
        self.create_directories()

    @property
    def output_directory(self) -> str:
        # make sure that tilde does not appear on the path.
        return os.path.expanduser(os.path.expandvars(self._output_directory))

    @property
    def temporary_directory(self) -> str:
        # make sure that tilde does not appear on the path.
        return os.path.expanduser(os.path.expandvars(self._temporary_directory))

    def create_directories(self) -> None:
        # Exception is raised if self.temporary_directory already exists.
        os.makedirs(self.temporary_directory)
        self._tmp_dir_created = True

        # Exception is raised if self.output_directory already exists.
        os.makedirs(self.output_directory)
        self._output_dir_created = True

    def __del__(self) -> None:
        self.delete_directories(force=False)

    def delete_directories(self, force: bool = True) -> None:
        if self.delete_output_folder_after_terminate or force:
            if self._output_dir_created is False:
                raise ValueError("Failed to delete output dir: %s because autoPyTorch did not "
                                 "create it. Please make sure that the specified output dir does "
                                 "not exist when instantiating autoPyTorch."
                                 % self.output_directory)
            try:
                shutil.rmtree(self.output_directory)
            except Exception:
                if self._logger is not None:
                    self._logger.warning("Could not delete output dir: %s" %
                                         self.output_directory)
                else:
                    print("Could not delete output dir: %s" %
                          self.output_directory)

        if self.delete_tmp_folder_after_terminate or force:
            if self._tmp_dir_created is False:
                raise ValueError("Failed to delete tmp dir: % s because autoPyTorch did not "
                                 "create it. Please make sure that the specified tmp dir does not "
                                 "exist when instantiating autoPyTorch."
                                 % self.temporary_directory)
            try:
                shutil.rmtree(self.temporary_directory)
            except Exception:
                if self._logger is not None:
                    self._logger.warning("Could not delete tmp dir: %s" % self.temporary_directory)
                else:
                    print("Could not delete tmp dir: %s" % self.temporary_directory)


class Backend(object):
    """Utility class to load and save all objects to be persisted.
    These are:
    * start time of auto-pytorch
    * true targets of the ensemble
    """

    def __init__(self, context: BackendContext):
        self.logger = logging.get_logger(__name__)
        self.context = context

        # Create the temporary directory if it does not yet exist
        try:
            os.makedirs(self.temporary_directory)
        except Exception:
            pass
        # This does not have to exist or be specified
        if self.output_directory is not None:
            if not os.path.exists(self.output_directory):
                raise ValueError("Output directory %s does not exist." % self.output_directory)

        self.internals_directory = os.path.join(self.temporary_directory, ".autoPyTorch")
        self._make_internals_directory()

    @property
    def output_directory(self) -> str:
        return self.context.output_directory

    @property
    def temporary_directory(self) -> str:
        return self.context.temporary_directory

    def _make_internals_directory(self) -> None:
        try:
            os.makedirs(self.internals_directory)
        except Exception as e:
            self.logger.debug("_make_internals_directory: %s" % e)
            pass

    def _get_start_time_filename(self, seed: Union[str, int]) -> str:
        if isinstance(seed, str):
            seed = int(seed)
        return os.path.join(self.internals_directory, "start_time_%d" % seed)

    def save_start_time(self, seed: str) -> str:
        self._make_internals_directory()
        start_time = time.time()

        filepath = self._get_start_time_filename(seed)

        if not isinstance(start_time, float):
            raise ValueError("Start time must be a float, but is %s." % type(start_time))

        if os.path.exists(filepath):
            raise ValueError(
                "{filepath} already exist. Different seeds should be provided for different jobs."
            )

        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(filepath), delete=False) as fh:
            fh.write(str(start_time))
            tempname = fh.name
        os.rename(tempname, filepath)

        return filepath

    def load_start_time(self, seed: int) -> float:
        with open(self._get_start_time_filename(seed), 'r') as fh:
            start_time = float(fh.read())
        return start_time

    def get_smac_output_directory(self) -> str:
        return os.path.join(self.temporary_directory, 'smac3-output')

    def get_smac_output_directory_for_run(self, seed: int) -> str:
        return os.path.join(
            self.temporary_directory,
            'smac3-output',
            'run_%d' % seed
        )

    def get_smac_output_glob(self, smac_run_id: Union[str, int] = 1) -> str:
        return os.path.join(
            glob.escape(self.temporary_directory),
            'smac3-output',
            'run_%s' % str(smac_run_id),
        )

    def _get_targets_ensemble_filename(self) -> str:
        raise NotImplementedError()

    def save_targets_ensemble(self, targets: np.ndarray) -> str:
        raise NotImplementedError()

    def load_targets_ensemble(self) -> np.ndarray:
        raise NotImplementedError()

    def _get_datamanager_pickle_filename(self) -> str:
        raise NotImplementedError()

    def get_done_directory(self) -> str:
        return os.path.join(self.internals_directory, 'done')

    def note_numrun_as_done(self, seed: int, num_run: int) -> None:
        done_directory = self.get_done_directory()
        os.makedirs(done_directory, exist_ok=True)
        done_path = os.path.join(done_directory, '%d_%d' % (seed, num_run))
        with open(done_path, 'w'):
            pass

    def get_model_dir(self) -> str:
        return os.path.join(self.internals_directory, 'models')

    def get_cv_model_dir(self) -> str:
        return os.path.join(self.internals_directory, 'cv_models')

    def get_model_path(self, seed: int, idx: int, budget: float) -> str:
        return os.path.join(self.get_model_dir(),
                            '%s.%s.%s.model' % (seed, idx, budget))

    def get_cv_model_path(self, seed: int, idx: int, budget: float) -> str:
        return os.path.join(self.get_cv_model_dir(),
                            '%s.%s.%s.model' % (seed, idx, budget))

    def save_model(self, model: BasePipeline, filepath: str) -> None:
        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(model, fh, -1)
            tempname = fh.name

        os.rename(tempname, filepath)

    def list_all_models(self, seed: int) -> List[str]:
        model_directory = self.get_model_dir()
        if seed >= 0:
            model_files = glob.glob(
                os.path.join(glob.escape(model_directory), '%s.*.*.model' % seed)
            )
        else:
            model_files = os.listdir(model_directory)
            model_files = [os.path.join(model_directory, model_file)
                           for model_file in model_files]

        return model_files

    def load_all_models(self, seed: int) -> Dict[Tuple[int, int, float], BasePipeline]:
        model_files = self.list_all_models(seed)
        models = self.load_models_by_file_names(model_files)
        return models

    def load_models_by_file_names(self, model_file_names: List[str]) -> Dict[Tuple[int, int, float], BasePipeline]:
        models = dict()

        for model_file in model_file_names:
            # File names are like: {seed}.{index}.{budget}.model
            if model_file.endswith('/'):
                model_file = model_file[:-1]
            if not model_file.endswith('.model') and \
                    not model_file.endswith('.model'):
                continue

            basename = os.path.basename(model_file)

            basename_parts = basename.split('.')
            seed = int(basename_parts[0])
            idx = int(basename_parts[1])
            budget = float(basename_parts[2])

            models[(seed, idx, budget)] = self.load_model_by_seed_and_id_and_budget(
                seed, idx, budget)

        return models

    def load_models_by_identifiers(self, identifiers: List[Tuple[int, int, float]]
                                   ) -> Dict:
        models = dict()

        for identifier in identifiers:
            seed, idx, budget = identifier
            models[identifier] = self.load_model_by_seed_and_id_and_budget(
                seed, idx, budget)

        return models

    def load_model_by_seed_and_id_and_budget(self, seed: int,
                                             idx: int,
                                             budget: float
                                             ) -> BasePipeline:
        model_directory = self.get_model_dir()

        model_file_name = '%s.%s.%s.model' % (seed, idx, budget)
        model_file_path = os.path.join(model_directory, model_file_name)
        with open(model_file_path, 'rb') as fh:
            return pickle.load(fh)

    def load_cv_models_by_identifiers(self, identifiers: List[Tuple[int, int, float]]
                                      ) -> Dict:
        models = dict()

        for identifier in identifiers:
            seed, idx, budget = identifier
            models[identifier] = self.load_cv_model_by_seed_and_id_and_budget(
                seed, idx, budget)

        return models

    def load_cv_model_by_seed_and_id_and_budget(self,
                                                seed: int,
                                                idx: int,
                                                budget: float
                                                ) -> BasePipeline:
        model_directory = self.get_cv_model_dir()

        model_file_name = '%s.%s.%s.model' % (seed, idx, budget)
        model_file_path = os.path.join(model_directory, model_file_name)
        with open(model_file_path, 'rb') as fh:
            return pickle.load(fh)

    def get_ensemble_dir(self) -> str:
        raise NotImplementedError()

    def _get_prediction_output_dir(self, subset: str) -> str:
        return os.path.join(self.internals_directory,
                            'predictions_%s' % subset)

    def get_prediction_output_path(self, subset: str,
                                   automl_seed: Union[str, int],
                                   idx: int,
                                   budget: float
                                   ) -> str:
        output_dir = self._get_prediction_output_dir(subset)
        # Make sure an output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return os.path.join(output_dir, 'predictions_%s_%s_%s_%s.npy' %
                            (subset, automl_seed, idx, budget))

    def save_predictions_as_npy(self,
                                predictions: np.ndarray,
                                filepath: str
                                ) -> None:
        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(predictions.astype(np.float32), fh, -1)
            tempname = fh.name
        os.rename(tempname, filepath)

    def save_predictions_as_txt(self,
                                predictions: np.ndarray,
                                subset: str,
                                idx: int, precision: int,
                                prefix: Optional[str] = None) -> None:
        # Write prediction scores in prescribed format
        filepath = os.path.join(
            self.output_directory,
            ('%s_' % prefix if prefix else '') + '%s_%s.predict' % (subset, str(idx)),
        )

        format_string = '{:.%dg} ' % precision
        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(
                filepath), delete=False) as output_file:
            for row in predictions:
                if not isinstance(row, np.ndarray) and not isinstance(row, list):
                    row = [row]
                for val in row:
                    output_file.write(format_string.format(float(val)))
                output_file.write('\n')
            tempname = output_file.name
        os.rename(tempname, filepath)

    def write_txt_file(self, filepath: str, data: str, name: str) -> None:
        with lockfile.LockFile(filepath):
            with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(
                    filepath), delete=False) as fh:
                fh.write(data)
                tempname = fh.name
            os.rename(tempname, filepath)
            self.logger.debug('Created %s file %s' % (name, filepath))
