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

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.datasource import Datasource, Datafetcher
from dltb.datasource import argparse as DatasourceArgparse
from dltb.util.image import imshow

# logging
LOG = logging.getLogger(__name__)


def output_info(datasource: Datasource) -> None:
    """Output information on the given :py:class:`Datasource`.
    """
    print(f"Datasource: {datasource}")
    print(f"Length: {len(datasource)}")
    print(f"Sized: {isinstance(datasource, Sized)}")
    print(f"Iterable: {isinstance(datasource, Iterable)}")
    print(f"Sequence: {isinstance(datasource, Sequence)}")


def output_data(iterable, args) -> None:
    """Process a sequence of data (pairs, data objects, or data batches)
    and output progress information.
    """
    erase_to_end_of_line = '\033[K'

    try:
        start = time.time()
        last = start
        count = 0
        index = 0
        for index, data in enumerate(iterable):
            now = time.time()
            time_info = (f"[{(now-last)*1000:5.1f}ms], "
                         f"average: {count/(now-start):.2f}")
            if isinstance(data, tuple):
                # pairs
                count += 1
                print(f"dl-datasource[{index}]: "
                      f"pair: {str(data[0].shape)}/{str(data[1].shape)}: "
                      f"{str(data[2]):7} "
                      f"{time_info} pairs per second",
                      erase_to_end_of_line, end='\r')
            elif data.is_batch:
                count += len(data)
                print(f"dl-datasource[{index}:{count-len(data)}-{count}]: "
                      f"batch[{len(data)}] of {type(data)}: "
                      f"{time_info} images per second",
                      erase_to_end_of_line, end='\r')
                if args.show:
                    imshow(data.visualize(size=(200, 200)), blocking=False)
            else:
                count += 1
                print(f"dl-datasource[{index}]: "
                      f"type(data) of shape {str(data.shape):20} "
                      f"{time_info} images per second",
                      erase_to_end_of_line, end='\r')
                if args.show:
                    imshow(data.visualize(), blocking=False)
            last = now
    except KeyboardInterrupt:
        print("Interrupted.", erase_to_end_of_line)

    print(f"dl-datasource: read {count} items" +
          (f" in {index+1} batches of size {args.batch}"
           if args.batch else "") +
          f" in {(last-start):.1f} seconds.",
          erase_to_end_of_line)


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
    parser.add_argument('--pairs', action='store_true', default=False,
                        help='enumerate (labeled) pairs (if available)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the data (if possible)')
    ToolboxArgparse.add_arguments(parser)
    DatasourceArgparse.prepare(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    datasource = DatasourceArgparse.datasource(parser, args)
    if datasource is None:
        logging.error("No datasource was specified.")
        return

    output_info(datasource)

    if args.fetcher:
        fetcher = Datafetcher(datasource, batch_size=args.batch)
        fetcher.reset()  # FIXME[hack]
        iterable = fetcher

    elif args.pairs:
        iterable = datasource.pairs()

    elif isinstance(datasource, Sequence):
        if args.batch:
            iterable = datasource.batches(size=args.batch)
        else:
            iterable = datasource
    else:
        iterable = None

    if iterable is not None:
        output_data(iterable, args)


if __name__ == "__main__":
    main()
