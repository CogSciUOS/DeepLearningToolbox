#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line tool for inspecting datasets.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import time
import logging
import argparse
from collections.abc import Sized, Iterable, Sequence

# third party imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from dltb.datasource import Datasource, Datafetcher
from dltb.datasource import argparse as DatasourceArgparse
# import dltb.thirdparty.datasource as _
# import dltb.thirdparty.keras.datasource as _

# logging
LOG = logging.getLogger(__name__)


def output_info(datasource: Datasource) -> None:
    print(f"Datasource: {datasource}")
    print(f"Length: {len(datasource)}")
    print(f"Sized: {isinstance(datasource, Sized)}")
    print(f"Iterable: {isinstance(datasource, Iterable)}")
    print(f"Sequence: {isinstance(datasource, Sequence)}")


def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description="Activation extraction from "
                                "layers of a neural network")
    parser.add_argument('--fetcher', action='store_true', default=False,
                        help='use a fetcher to traverse the datasource')
    parser.add_argument('--batch', type=int,
                        help='process data in batches of given size')
    DatasourceArgparse.prepare(parser)
    args = parser.parse_args()

    datasource = DatasourceArgparse.datasource(parser, args)
    if datasource is None:
        logging.error("No datasource was specified.")
        return

    output_info(datasource)

    erase_to_end_of_line = '\033[K'

    if args.fetcher:
        fetcher = Datafetcher(datasource, batch_size=args.batch)
        fetcher.reset()  # FIXME[hack]
        iterable = fetcher

    elif isinstance(datasource, Sequence):
        if args.batch:
            iterable = datasource.batches(size=args.batch)
        else:
            iterable = datasource
    else:
        iterable = None

    if iterable is not None:
        try:
            start = time.time()
            last = start
            count = 0
            for index, data in enumerate(iterable):
                now = time.time()
                if data.is_batch:
                    count += len(data)
                    print(f"dl-datasource[{index}]: "
                          f"batch[{len(data)}] of {type(data)}: "
                          f"[{(now-last)*1000:5.1f}ms], "
                          f"average: {count/(now-start):.2f} "
                          f"images per second",
                          erase_to_end_of_line, end='\r')
                else:
                    count += 1
                    print(f"dl-datasource[{index}]: "
                          f"type(data) of shape {str(data.shape):20} "
                          f"[{(now-last)*1000:5.1f}ms], "
                          f"average: {count/(now-start):.2f} "
                          f"images per second",
                          erase_to_end_of_line, end='\r')
                last = now
        except KeyboardInterrupt:
            print("Interrupted.", erase_to_end_of_line)

        print(f"dl-datasource: read {count} items " +
              (f"in {index+1} batches of size {args.batch}"
               if args.batch else "") +
              f"in {(last-start):.1f} seconds.",
              erase_to_end_of_line)


if __name__ == "__main__":
    main()
