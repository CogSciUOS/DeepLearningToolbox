#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line tool for inspecting datasets.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import logging
import argparse

# third party imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from dltb.datasource import Datasource, Datafetcher
from dltb.datasource import argparse as DatasourceArgparse
import datasource.predefined

# logging
LOG = logging.getLogger(__name__)


def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description="Activation extraction from "
                                "layers of a neural network")
    DatasourceArgparse.prepare(parser)
    args = parser.parse_args()

    datasource = DatasourceArgparse.datasource(args)
    if datasource is None:
        logging.error("No datasource was specified.")
        return

    batch_size = 128
    fetcher = Datafetcher(datasource, batch_size=batch_size)
    fetcher.reset() # FIXME[hack]
    try:
        samples = len(datasource)
        # Here we could:
        #  np.memmap(filename, dtype='float32', mode='w+',
        #            shape=(samples,) + network[layer].output_shape[1:])
        index = 0
        for batch in fetcher:
            print("dl-activation: batch:", type(batch.array),
                  len(batch), len(batch.array))
            print([data.array.shape for data in batch])
            print("dl-activation: indices:", batch[0].index, batch[-2].index)
            print("dl-activation: batch finished.")
    except InterruptedError:
        print("Interrupted.")


if __name__ == "__main__":
    main()
