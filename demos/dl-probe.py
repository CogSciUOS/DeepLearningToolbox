#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for extracting activation values from a network.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import logging
import argparse

# third party imports
import numpy as np

# toolbox imports


from dltb.tool.probe import ProbeClassifier, PhdLabWrapper



def main():
    """The main program.
    """
    print("Hello")
    probe = ProbeClassifier()

    print("Hello2")
    PhdLabWrapper. \
        init_phd_lab('/net/store/cv/users/krumnack/projects/github/phd-lab')

    model = 'resnet18_XXS'
    dataset = 'Cifar10',
    resolution = '32'
    model_name = PhdLabWrapper.parse_model(model, (32, 32, 3), 10)
    print("Canonical model name:", model_name)
    # ResNet18_XXS

    # 
    #model_name = self.parse_model(model, (32, 32, 3), 10)
    #directory = os.path.join(args.folder,
    #                         f'{model_name}_{dataset}_{resolution}')
    #self._arguments = PseudoArgs(model_name=model_name,
    #                             folder=directory, mp=args.mp)
    # latent_datasets/ResNet18_XXS_Cifar10_32

    phd_lab = PhdLabWrapper(config_file='test.json')
    print(phd_lab.activations_directory)

    # print("List of all numpy files in the activations_directory: ",
    #       PhdLabWrapper.get_all_npy_files(phd_lab.activations_directory),
    #       "\n")

    # train_set, test_set = \
    #     PhdLabWrapper.obtain_all_dataset(phd_lab.activations_directory)
    # print("List of all files in the training set:", train_set, "\n")
    # print("List of all files in the test set:", test_set, "\n")    

    # train_data, train_label = train_set[0]
    # test_data, test_label = test_set[0]
    # model = probe.train_model_from_pickle_files(train_data, train_label)
    # print("Training accuracy: ",
    #       probe.model_accuracy_from_pickle_files(model, train_data,
    #                                              train_label))
    # print("Test accuracy: ",
    #     probe.model_accuracy_from_pickle_files(model, test_data, test_label))

    probe.run(phd_lab.activations_directory)    

if __name__ == "__main__":
    main()

