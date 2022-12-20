"""Datasource interface to the CIFAR dataset.
"""
# FIXME[new]: this is not yet implemented.  This file holds just
# some code snippets from
# https://github.com/MINGUKKANG/Adversarial-AutoEncoder.git
# file: data_utils.py
#
# The new implementation may be based on ./mnist.py

# standard imports
import logging
import tarfile

# thirdparty imports
import numpy as np

# toolbox imports
from dltb import config
from dltb.util.download import download
from dltb.types import Path, Pathlike, as_path
from dltb.datasource import Imagesource
from dltb.datasource.array import LabeledArray
from dltb.tool.classifier import ClassScheme

# logging
LOG = logging.getLogger(__name__)


class CIFAR(LabeledArray, Imagesource):


    def _prepare(self) -> None:
        if self.type == "CIFAR_10":
            self.url = "https://www.cs.toronto.edu/~kriz/"
            self.debug = 1
            self.n_train_images = 50000
            self.n_test_images = 10000
            self.n_channels = 3
            self.size = 32
            self.CIFAR_10_filename = ["cifar-10-python.tar.gz"]

        # download data
        if self.type == "CIFAR_10":
            for i in self.CIFAR_10_filename:
                filepath = os.path.join("./Data", i)
                self.maybe_download(i,filepath)
                self.filepath_holder.append(filepath)

    # this code is from
    # https://github.com/melodyguan/enas/blob/master/src/cifar10/data_utils.py
    def extract_cifar_data(self,filepath, train_files,n_images):
        images, labels = [], []
        for file_name in train_files:
            full_name = os.path.join(filepath, file_name)
            with open(full_name, mode = "rb") as finp:
                data = pickle.load(finp, encoding = "bytes")
                batch_images = data[b'data']
                batch_labels = np.array(data[b'labels'])
                images.append(batch_images)
                labels.append(batch_labels)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        one_hot_encoding = np.zeros((n_images, 10))
        one_hot_encoding[np.arange(n_images), labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, 10])
        images = np.reshape(images, [-1, 3, 32, 32])
        images = np.transpose(images, [0, 2, 3, 1])

        return images, one_hot_encoding

    def extract_cifar_data_(self,filepath, num_valids=5000):
        print("Reading data")
        with tarfile.open(filepath, "r:gz") as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, "./Data")
        images, labels = {}, {}
        train_files = [
            "./cifar-10-batches-py/data_batch_1",
            "./cifar-10-batches-py/data_batch_2",
            "./cifar-10-batches-py/data_batch_3",
            "./cifar-10-batches-py/data_batch_4",
            "./cifar-10-batches-py/data_batch_5"]
        test_file = ["./cifar-10-batches-py/test_batch"]
        images["train"], labels["train"] = self.extract_cifar_data("./Data", train_files,self.n_train_images)

        if num_valids:
            images["valid"] = images["train"][-num_valids:]
            labels["valid"] = labels["train"][-num_valids:]

            images["train"] = images["train"][:-num_valids]
            labels["train"] = labels["train"][:-num_valids]
        else:
            images["valid"], labels["valid"] = None, None

        images["test"], labels["test"] = self.extract_cifar_data("./Data", test_file,self.n_test_images)
        return images, labels
