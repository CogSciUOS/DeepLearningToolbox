"""Probe classifiers.

Probe classifiers can be used as an estimate how good data has been
transformed to perform a classification task.

taken from:
phd_lab/train_probes.py 
phd_lab/experiments/probe_training.py


"""

# standard imports
from typing import List, Tuple
from itertools import product
import os
import sys
import json

# third-party modules
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel
import pickle
import pandas as pd
from multiprocessing import Pool

# https://www.attrs.org/en/stable/
# Classes Without Boilerplate
from attr import attrs, attrib

# https://joblib.readthedocs.io/en/latest/
from joblib import Memory

# toolbox imports
from .tool import IterativeTool


# The idea of this class seems to be to simply provide some arguments
# specifiying an experiment.
#
# Example:
#   folder:      Contains latent representations for a model+dataset
#                e.g., latent_datasets/ResNet18_XXS_Cifar10_32/
#   save_path:   Instrospection data, like training performance
#                probe_performances, etc.
#                e.g., logs/ResNet18_XXS/Cifar10_32/myreceptive
@attrs(auto_attribs=True, slots=True)
class PseudoArgs:  # why Pseudo
    """The pseudo args configuring the training of a probe.
    Args:
        model_name: str
            The name of the model                    
        folder: str
            The folder containing the latent representation
        mp: int
            Number of processes to use of multiprocessing
        overwrite: bool
            Overwrite existing results
    """
    model_name: str
    folder: str
    mp: int
    save_path: str = attrib(init=False)
    overwrite: bool = False

    def __attrs_post_init__(self):
        # attr initialization magic:

        # the `model_pointer.txt` file contains the path to a directory
        # to which model specific results should be stored
        filename = os.path.join(self.folder, "model_pointer.txt")
        self.save_path = open(filename, "r").read()


class ProbeClassifier(IterativeTool):
    """A :py:class:`ProbeClassifier` is a simple probe classifier that is
    used to estimate how hard a classification task is.  This is
    expressed by the performace the classifier can reach on the task:
    if it can reach a high performance the task is deemed easy while a
    bad performance indicates a hard task.


    Properties
    ----------
    probe_performance_savefile: str
        Name of the CSV file to which probe performance results
        are written.

    
    """
    probe_performance_savefile: str = 'probe_performances.csv'

    _overwrite: bool = False
    

    def __init__(self, prefix: str = None, config: str = None,
                 **kwargs) -> None:
        """
        Arguments
        ---------
        prefix: str
            An optional prefix that will be prepended to the name
            of the probe performance file.
            
        """
        super().__init__(**kwargs)

        if prefix is not None:
            self.probe_performance_savefile = \
                prefix + '_' + self.probe_performance_savefile

    def run(self, activations_directory: str,
            overwrite: bool = False, processors: int = None) -> None:
        """Execute model training on latent representations

        Arguments
        ---------
        args:
            The configuration of the training as PseudoArgs object
        """

        #
        # Sanity checks
        #
        if os.path.exists(self.probe_performance_savefile):
            print('Detected existing results', self.probe_performance_savefile)
            if overwrite:
                print("overwriting is enabled. Training will continue "
                      "and previous results will be overwritten.")
            else:
                print("overwriting is disabled, stopping...")
                return

        # trainset: List[Tuple[train_activation_file, train_label_file]]
        # testset: List[Tuple[test_activation_file, test_label_file]]
        train_set, eval_set = \
            PhdLabWrapper.obtain_all_dataset(activations_directory)
        # both list should contain the same number of elements, listing
        # all layers of interest
        if len(train_set) != len(eval_set):
            raise FileNotFoundError("Number of training sets "
                                    f"({len(train_set)}) does not"
                                    "match the number of test sets "
                                    f"({len(eval_set)})."
                                    "Make sure the datas has been extracted "
                                    "correctly. Consider rerunning extraction")

        # names: list of names of the activation files
        # train_accuracies: list of accuracy values for the training data
        # test_accuracies: list of accuracy values for the test data
        self._names, self._train_accuracies, self._test_accuracies = [], [], []

        if processors:
            self._run_parallel(train_set, eval_set, processors)
        else:
            self._run_sequential(train_set, eval_set)

    def _run_sequential(self, train_set, eval_set) -> None:
        """Run the probe classifier on the given training and
        evaluation set.
        """
        print('Multiprocessing is disabled starting training...')
        for train_data, eval_data in zip(train_set, eval_set):
            name, train_accuracy, test_accurracy = \
                self._probe_pickle_files(train_data, eval_data)
            self._names.append(name)
            self._train_accuracies.append(train_accuracy)
            self._test_accuracies.append(test_accurracy)
            self._write_results_to_file()

    def _run_parallel(self, train_set, eval_set, processors: int) -> None:
        # FIXME[old/todo]: parallel processing (using multiple processes)
        # the `Parallel` class is from the third party module `joblib`
        print('Multiprocessing is enabled starting training...')
        for train_data, eval_data in zip(train_set, eval_set):
            args.append((os.path.basename(train_data[0][:-2]),
                         train_data, eval_data))

        p = Parallel(n_jobs=processors, verbose=1000)
        results = p(delayed(self._probe_pickle_files)(arg) for arg in args)
        for name, train_accuracy, test_accurracy in results:
            self._names.append(name)
            self._train_accuracies.append(train_accuracy)
            self._test_accuracies.append(test_accurracy)
    
        self._write_results_to_file()

    def _probe_pickle_files(self, train_set: Tuple[str, str],
                            eval_set: Tuple[str, str]
                            ) -> Tuple[str, float, float]:
        """Perform probe training and evaluation on training and test data.

        Arguments
        ---------
        train_set: Tuple[str, str]
            File names for training data and labels.
        eval_set:  Tuple[str, str]
            File names for test data and labels.
        """
        name = os.path.basename(train_set[0][:-2])
        print(f"Training model for '{name}'")
        model = self.train_model_from_pickle_files(*train_set)
        print(f"Obtaining metrics for '{name}'")
        train_accuracy = \
            self.model_accuracy_from_pickle_files(model, *train_set)
        test_accuracy = \
            self.model_accuracy_from_pickle_files(model, *eval_set)
        print(os.path.basename(train_set[0]))
        print(f"Accuracy for '{name}': "
              f"train={train_accuracy:.3f}, test={test_accuracy:.3f}")
        return name, train_accuracy, test_accuracy

    def _write_results_to_file(self) -> None:
        """Write currently computed values into a file.
        """
        pd.DataFrame.from_dict(
            {
                'name': self._names,
                'train_acc': self._test_accuracies,
                'eval_acc': self._test_accuracies
            }
        ).to_csv(self.probe_performance_savefile, sep=';')

    def train_model(self, data: np.ndarray,
                    labels: np.ndarray) -> LogisticRegressionModel:
        """Train a logistic regression model on latent representations
        and labels from the original dataset.
        
        Arguments
        ---------
        data: np.ndarray
            The data provided as numpy array.
        labels: np.ndarray
            The corresponding labels as numpy array.

        Result
        ------
        model: LogisticRegressionModel:
            A logistic regression model fitted on the provided data
        """
        return fit_with_cache(data, labels)

    def model_accuracy(self, model: LogisticRegressionModel,
                       data: np.ndarray, labels: np.ndarray) -> float:
        """Obtain the probe performance from a fitted logistic regression
        model.

        Arguments
        ---------
        model: LogisticRegressionModel
            The (already fitted) model.
        data: numpy.ndarray
            The evaluation data
        labels: numpy.ndarray
            The evaluation labels

        Result
        ------
        accuracy: float
            The accuracy value (between 0 and 1).

        """
        print('Evaluating with data of shape', data.shape)
        preds = model.predict(data)
        return accuracy_score(labels, preds)

    #
    # working with pickle files
    #

    def train_model_from_pickle_files(self, data_path: str, labels_path: str
                                      ) -> LogisticRegressionModel:
        """Train a logistic regression model on latent representations
        and labels from the original dataset.
        
        Arguments
        ---------
        data_path: str
            Fully qualitified name of the data file. The file is
            expected to contain data as a pickled numpy array.
        labels_path: str
            Fully qualitified name for the class labels file. The file is
            expected to contain labels as a pickled numpy array.

        Result
        ------
        model: LogisticRegressionModel:
            A logistic regression model fitted on the provided data
        """
        print('Loading training data from', data_path)
        data, labels = \
            self.load_data_and_labels_from_pickle_files(data_path, labels_path)
        print('Training data obtained with shape', data.shape)
        return self.train_model(data, labels)

    def model_accuracy_from_pickle_files(self, model: LogisticRegressionModel,
                                         data_path: str,
                                         label_path: str) -> float:
        """Obtain the probe performance from a fitted logistic regression
        model.

        Arguments
        ---------
        model: LogisticRegressionModel
            The (already fitted) model.
        data_path: str
            The path to the evaluation data.
        label_path: str
            The path to the evaluation labels.

        Result
        ------
        accuracy: float
            The accuracy value (between 0 and 1).

        """
        data, labels = \
            self.load_data_and_labels_from_pickle_files(data_path, label_path)
        print('Loaded data:', data_path)
        return self.model_accuracy(model, data, labels)

    #
    # private
    #

    @staticmethod
    def _unpickle_file(filename: str) -> np.ndarray:
        """Load all data from a pickle file batch-wise.
        Data are stored in the file batchwise, by repeatedly writing
        to the file. This function iterate over all chunks found in
        the file.

        Arguments
        ---------
        filename: str
            The name of the pickle file.

        Result
        ------
        array:
            the data as numpy-array
        """
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def _load_pickle_file(self, filename: str) -> np.ndarray:
        """Load a large pickle file batch-wise  and append the data.

        Arguments
        ---------
        filename: str
            The filenam to load the data from.

        Result
        ------
        array:
            The data as numpy array
        """
        return np.vstack([batch for batch in self._unpickle_file(filename)])

    def load_data_and_labels_from_pickle_files(self, data_path: str,
            label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load the dataset and labels ready for training.

        Arguments
        ---------
        data_path:
            The path the data is stored
        label_path:
            The path to the label data

        Result
        ------
        Training data and labels, ready for training.
        """
        return (self._load_pickle_file(data_path),
                np.squeeze(self._load_pickle_file(label_path)))





def fit_with_cache(data: np.ndarray, labels: np.ndarray):
    model = LogisticRegressionModel(multi_class='multinomial', n_jobs=12,
                                    solver='saga', verbose=True)
    model = model.fit(data, labels)
    return model


class PhdLabWrapper:

    phd_lab_directory: str = None

    @classmethod
    def init_phd_lab(cls, directory: phd_lab_directory) -> None:
        if cls.phd_lab_directory is not None:
            raise RuntimeError("Cannot initialize PhdLabWrapper "
                               "a second time!")
        cls.phd_lab_directory = directory

        memcache = os.path.join(cls.phd_lab_directory, 'memcache')
        if not os.path.exists(memcache):
            os.makedirs(memcache)
        memory = Memory(".memcache", verbose=True)

        global fit_with_cache
        fit_with_cache = memory.cache(fit_with_cache)
    
    @staticmethod
    def create_pseudo_args() -> PseudoArgs:
        args = PseudoArgs(model_name='ResNet18',
                          folder=PhdLabWrapper.phd_lab_directory)

    
    #
    # static methods
    #

    
    @staticmethod
    def parse_model(model_name, shape, num_classes) ->  str:
        """Get the name of the model
        Args:
        model_name:
           factory name of the model
        shape:
           input shape (may be any 2 tupe of ints)
        num_classes:
            the number of classes of the dataset (may be any integer)
        Returns:
        the actual name of the model
        """
        try:
            if not PhdLabWrapper.phd_lab_directory in sys.path:
                sys.path.insert(0, PhdLabWrapper.phd_lab_directory)
            # from module import symbol
            from phd_lab.experiments.utils.config import MODEL_REGISTRY as models
            model = models.__dict__[model_name](input_size=shape,
                                                num_classes=num_classes)
        except KeyError:
            raise NameError("%s doesn't exist." % model_name)
        return model.name

    #
    # Configurations
    #

    def __init__(self, config_file: str = None) -> None:
        """
        Arguments
        ---------
        config_file: str
            Name of a configuration file (in JSON format).
            If not `None`, configuration data will be read from that
            file. Configuration attributes of interest are 'model',
            'dataset', and 'resolution'.
        """
        if not os.path.isabs(config_file) and not os.path.isfile(config_file):
            for directory in (self.phd_lab_directory,
                              os.path.join(self.phd_lab_directory, 'configs')):
                filename = os.path.join(directory, config_file)
                if os.path.isfile(filename):
                    config_file = filename
                    break
        self._config = json.load(open(config_file, 'r'))
        model = self._config['model']
        model = model[0] if isinstance(model, list) else model
        dataset = self._config['dataset']
        dataset = dataset[0] if isinstance(dataset, list) else dataset
        resolution = self._config["resolution"]
        resolution = resolution[0] if isinstance(resolution, list) else resolution
        model_name = self.parse_model(model, (32, 32, 3), 10)
        
        self._activations_directory = \
            os.path.join(self.phd_lab_directory, 'latent_datasets',
                         f'{model_name}_{dataset}_{resolution}')

    @property
    def activations_directory(self):
        return self._activations_directory

    def run(self, probe_classifier: ProbeClassifier) -> None:
        # FIXME[hack]:
        args = PseudoArgs(model_name='ResNet18',
                          folder='folder')

        #
        # run the main program
        #
        if self.config is None:
            args = self.create_pseudo_args('ResNet18')
            self.main(args)
        else:
            # run the main program for every conmbination of
            # model x dataset x resolution
            self._config = json.load(open(config, 'r'))
            models = self._config['model']
            datasets = self._config['dataset']
            resolutions = self._config["resolution"]
            for (model, dataset, resolution) in \
                    itertools.product(models, datasets, resolutions):
                model_name = self.parse_model(model, (32, 32, 3), 10)
                directory = os.path.join(args.folder,
                                         f'{model_name}_{dataset}_{resolution}')
                self._arguments = PseudoArgs(model_name=model_name,
                                             folder=directory, mp=args.mp)
                print(pargs)
                self.main(pargs)


    def obtain_all_dataset(folder: str) -> Tuple[List[Tuple[str, str]],
                                                 List[Tuple[str, str]]]:
        """Build the training and test datasets from all available latent
        representations.
        
        Arguments
        ---------
        folder:
            The folder containing labels and latent representations
        
        Result
        ------
        train_set:
            The list of all pairs of absolute filenames for the training set
            (pairing each training activation file with the training
            label file).
        test_set: 
            The list of all pairs of absolute filenames for the test set
            (pairing each test activation file with the test label file).
        """
        all_files = PhdLabWrapper.get_all_npy_files(folder)
        data, labels = PhdLabWrapper._separate_labels_from_data(all_files)
        train_data, train_label = \
            (PhdLabWrapper._filter_files_by_string_key(data, 'train-'),
             PhdLabWrapper._filter_files_by_string_key(labels, 'train-'))
        eval_data, eval_label = \
            (PhdLabWrapper._filter_files_by_string_key(data, 'eval-'),
             PhdLabWrapper._filter_files_by_string_key(labels, 'eval-'))
        train_set = [elem for elem in product(train_data, train_label)]
        eval_set = [elem for elem in product(eval_data, eval_label)]
        return train_set, eval_set

    def get_all_npy_files(folder: str) -> List[str]:
        """Get the names of all numpy files stored in a folder. Numpy
        files are identified by the suffix '.p'.

        Arguments
        ---------
        folder:
            The target folder.
        
        Result
        ------
        The npy-files as full filepaths
        """
        all_files = os.listdir(folder)
        filtered_files = \
            PhdLabWrapper._filter_files_by_string_key(all_files, '.p')
        full_paths = [os.path.join(folder, file) for file in filtered_files]
        return full_paths

    def _filter_files_by_string_key(files: List[str], key: str) -> List[str]:
        """Filter files by a substring, returning only those filenames
        that contain key.

        Arguments
        ---------
        files:
            list of filepaths
        key:
            the substring the file most contain in order to be not filtered

        Result
        ------
            The filtered list of paths
        """
        return [file for file in files if key in file]

    def _separate_labels_from_data(files: List[str]) -> Tuple[List[str],
                                                              List[str]]:
        """Separate taining data files and label files.

        Arguments
        ---------
        files:
            The list of filenames to be separated.
 
        Result
        ------
        data_files: List[str]
            List of all data files (filename not containing the
            substring '-labels')
        label_files: List[str]
            List of all labels files (filename containing the
            substring '-labels')
        """
        data_files = [file for file in files if '-labels' not in file]
        label_file = [file for file in files if '-labels' in file]
        return data_files, label_file
