# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from Utils.utils import load_json_table
from prettytable import PrettyTable
import numpy
import os
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def get_statistics(in_tab_fn):
    tables = load_json_table(in_tab_fn)
    table_num = len(tables)

    row_num_population = []
    for tab_id in tables:
        row_num_population.append(len(tables[tab_id]))

    row_num_std = numpy.std(row_num_population)
    avg_row_num = sum(row_num_population) / len(row_num_population)

    col_num_population = []
    for tab_id in tables:
        if len(tables[tab_id][0]) == 1:
            print(tab_id)
        col_num_population.append(len(tables[tab_id][0]))

    col_num_std = numpy.std(col_num_population)
    avg_col_num = sum(col_num_population) / len(col_num_population)
    return table_num, avg_row_num, row_num_std, avg_col_num, col_num_std
    # results.add_row([table_num, row_num, row_num_std, col_num, col_num_std])
    # print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Input"
    )
    args = process_relative_path_config(parser.parse_args())
    results = PrettyTable()
    results.field_names = [
        "dataset",
        "table_num",
        "avg_row_num",
        "row_std",
        "avg_col_num",
        "col_num_std"
    ]
    in_dir = args.in_dir
    for fn in os.listdir(in_dir):
        if fn.endswith('.json'):
            print(fn)
            table_num, avg_row_num, row_num_std, avg_col_num, col_num_std = get_statistics(os.path.join(in_dir, fn))
            print([fn[:-5], table_num, avg_row_num, row_num_std, avg_col_num, col_num_std])
            results.add_row([fn[:-5], table_num,
                             "{:.1f}".format(avg_row_num),
                             "{:.1f}".format(row_num_std),
                             "{:.1f}".format(avg_col_num),
                             "{:.1f}".format(col_num_std)])

    print(results)
