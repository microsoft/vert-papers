# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from multiprocessing import Process, Queue, current_process, freeze_support
import queue
from TableAnnotator.Config.config_utils import process_config
from TableAnnotator.Detect.multi_subject_table_annotator import LinkingPark
import json
from TableAnnotator.Util.utils import InputTable
import argparse
import math
from datetime import datetime
import random
from tqdm import tqdm
import os


def merge_results(all_lines):
    all_table_out_lines = []
    for x in all_lines:
        all_table_out_lines += x
    return all_table_out_lines


# multi-process utils
def run(func, args):
    print('{} process running'.format(current_process().name))
    ans = func(*args)
    print('{} process ok'.format(current_process().name))
    return ans


def worker(task_queue, done_queue):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            func, args = task_queue.get_nowait()
            print('Start new process')
        except queue.Empty:
            print('Error')
            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            result = run(func, args)
            done_queue.put(result)


def detect_one_table(detector, params, table):
    output_tab = detector.detect_single_table(table,
                                              keep_N=params["keep_N"],
                                              alpha=params["alpha"],
                                              beta=params["beta"],
                                              gamma=params["gamma"],
                                              topk=params["topk"],
                                              init_prune_topk=params["init_prune_topk"],
                                              max_iter=params["max_iter"],
                                              min_final_diff=params["min_final_diff"],
                                              row_feature_only=params["row_feature_only"],
                                              min_property_threshold=params["min_property_threshold"])
    return output_tab


def slice_table(table, max_row_size=20):
    row_size = len(table)
    tables = []
    for i in range(math.ceil(row_size/max_row_size)):
        tables.append(table[i*max_row_size:(i+1)*max_row_size])
    return tables


def split_mini_tables(in_fn, max_row_size=10):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        all_mini_input_tabs = []
        for i, line in enumerate(fp):
            words = line.strip().split("\t")
            tab_id, table_str = words[0], words[1]
            big_table = json.loads(table_str.strip())
            tables = slice_table(big_table, max_row_size=max_row_size)
            for idx, mini_tab in enumerate(tables):
                # input_tab = InputTable(mini_tab, "{}||{}".format(tab_id, idx))
                # all_mini_input_tabs.append(input_tab)
                all_mini_input_tabs.append("{}\t{}\n".format("{}||{}".format(tab_id, idx), json.dumps(mini_tab)))
            # if i > 50:
            #     break
        return all_mini_input_tabs


def divide_mini_tables(all_mini_input_tabs, items_per_process):
    n_parts = math.ceil(len(all_mini_input_tabs) / items_per_process)
    part_tables = []
    for i in range(n_parts):
        part_tables.append(all_mini_input_tabs[i * items_per_process:(i + 1) * items_per_process])
    assert (len(all_mini_input_tabs) == sum([len(x) for x in part_tables]))
    return part_tables


def detect_multi_tables(params, in_fn, out_fn):
    t1 = datetime.now()
    detector = LinkingPark(params)
    t2 = datetime.now()
    model_built_time = (t2-t1).total_seconds()
    print("{} model built time: {} seconds".format(current_process().name, model_built_time))
    print("detector building done.")
    num = 0
    result_lines = []
    part_tables = []
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            words = line.strip().split('\t')
            tab_id = words[0]
            sub_tab = json.loads(words[1])
            part_tables.append(InputTable(sub_tab, tab_id))

    for input_tab in tqdm(part_tables):
        output_tab = detect_one_table(detector, params, input_tab)
        small_dump_tab = output_tab.dump_one_multisubj_tab()
        num += 1
        # if num % 100 == 0:
        #     print("predict {} mini tables...".format(num))
        result_lines.append("{}\n".format(json.dumps(small_dump_tab)))
    t3 = datetime.now()
    model_run_and_format_time = (t3-t2).total_seconds()
    print("{} model run and format time: {} seconds".format(current_process().name, model_run_and_format_time))
    with open(out_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines(result_lines)
    t4 = datetime.now()
    dump_time = (t4 - t3).total_seconds()
    print("{} predictions write time: {} seconds".format(current_process().name, dump_time))
    print("---------------")
    for time_name in detector.time_counter:
        print("{}\t{}\t{}".format(current_process().name, time_name, detector.time_counter[time_name]))
    print("---------------")


def parallel_processing_mini_tables(params,
                                    in_fn,
                                    dump_dir,
                                    items_per_process,
                                    max_row_size=20,
                                    number_process=10):

    t1 = datetime.now()
    print("[1] Split big table into multiple mini tables...")
    all_mini_input_tabs = split_mini_tables(in_fn, max_row_size=max_row_size)

    print("    Total mini_input_tables: {}".format(len(all_mini_input_tabs)))

    print("[2] Divide mini tables into multiple processes...")
    part_tables = divide_mini_tables(all_mini_input_tabs, items_per_process)
    print("    Total {} jobs".format(len(part_tables)))
    for i in range(len(part_tables)):
        with open(os.path.join(dump_dir, f"input_part_{i}.jsonl"), mode="w", encoding="utf-8") as fp:
            fp.writelines(part_tables[i])
    t_s = datetime.now()
    print("    Total split time: {}".format((t_s-t1).total_seconds()))

    tasks = []

    print("[3] Running each process ...")
    for i, x in enumerate(part_tables):
        tasks.append((detect_multi_tables, (params,
                                            os.path.join(dump_dir, f"input_part_{i}.jsonl"),
                                            os.path.join(dump_dir, f"output_part_{i}.jsonl"))))

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for task in tasks:
        task_queue.put(task)

    print('    Start {} processes'.format(number_process))
    processes = []
    # Start worker processes
    for i in range(number_process):
        processes.append(Process(target=worker, args=(task_queue, done_queue)))

    for p in processes:
        p.start()

    print("    Waiting each process's stop")
    for p in processes:
        p.join()

    t_e = datetime.now()
    print("    Total subprocesses running time: {}".format((t_e - t_s).total_seconds()))

    print("[4] Merging predictions ...")
    small_dump_tab_list = []

    for i in range(len(part_tables)):
        with open(os.path.join(dump_dir, f"output_part_{i}.jsonl"), mode="r", encoding="utf-8") as fp:
            for line in fp:
                small_dump_tab_list.append(json.loads(line.strip()))

    dump_min_tables(small_dump_tab_list, os.path.join(dump_dir, f"all_output_tables.jsonl"))
    t2 = datetime.now()
    print("    Total dump prediction time: {}".format((t2 - t_e).total_seconds()))
    print("Time Consumed: {}s\n".format((t2 - t1).total_seconds()))
    print("--------OVER--------")
    exit()


def load_sub_tables(fn):
    all_tables = dict()
    with open(fn, encoding="utf-8", mode="r") as fp:
        for line in fp:
            tab = json.loads(line.strip())
            tab_id = tab['tab_id'].split("||")[0]
            part_id = int(tab['tab_id'].split("||")[1])
            if tab_id not in all_tables:
                all_tables[tab_id] = []
            tab["tab_id"] = tab_id
            all_tables[tab_id].append((tab, part_id))
    re_all_tables = dict()
    for tab_id in all_tables:
        sorted_mini_tables = sorted(all_tables[tab_id], key=lambda x:x[1])
        re_all_tables[tab_id] = [x[0] for x in sorted_mini_tables]
    return re_all_tables


def dump_min_tables(small_dump_tab_list, out_fn):
    all_tables = dict()

    for tab in small_dump_tab_list:
        tab_id = tab['tab_id'].split("||")[0]
        part_id = int(tab['tab_id'].split("||")[1])
        if tab_id not in all_tables:
            all_tables[tab_id] = []
        all_tables[tab_id].append((tab, part_id))
    for tab_id in all_tables:
        sorted_mini_tables = sorted(all_tables[tab_id], key=lambda x: x[1])
        all_tables[tab_id] = [x[0] for x in sorted_mini_tables]

    with open(out_fn, mode="w", encoding="utf-8") as re_fp:
        num = 0
        for tab_id in all_tables:
            tab = all_tables[tab_id]
            re_tab = dict()
            re_tab["tab_id"] = tab_id
            re_tab["row_size"] = 0
            re_tab["col_size"] = tab[0]['col_size']
            re_tab["revisit_predict_entities"] = dict()
            re_tab["predict_entities"] = dict()
            re_tab["tab_pred_type"] = tab[0]["tab_pred_type"]
            re_tab["coarse_properties"] = tab[0]["coarse_properties"]
            re_tab["coarse_candid_entities_info"] = dict()
            offset = 0
            for sub_tab in tab:
                re_tab["row_size"] += sub_tab["row_size"]
                for row_col_id in sub_tab["revisit_predict_entities"]:
                    row_id, col_id = row_col_id[1:-1].split(', ')
                    re_tab["revisit_predict_entities"][str((offset+int(row_id), int(col_id)))] = sub_tab["revisit_predict_entities"][row_col_id]
                for row_col_id in sub_tab["predict_entities"]:
                    row_id, col_id = row_col_id[1:-1].split(', ')
                    re_tab["predict_entities"][str((offset+int(row_id), int(col_id)))] = sub_tab["predict_entities"][row_col_id]
                for row_col_id in sub_tab["coarse_candid_entities_info"]:
                    row_id, col_id = row_col_id[1:-1].split(', ')
                    re_tab["coarse_candid_entities_info"][str((offset+int(row_id), int(col_id)))] = sub_tab["coarse_candid_entities_info"][row_col_id]
                offset += sub_tab['row_size']
            re_fp.write("{}\n".format(json.dumps(re_tab)))
            num += 1
            if num % 10000 == 0:
                print("Dump {} lines...".format(num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_table_fn",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Input/Semtab21_round2_biotab_table.json")
    parser.add_argument("--dump_dir",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Output/round4_semtab_measure_speed_rm_cell_filter_keep_dis_rocksdb_single_process")
    parser.add_argument("--topk",
                        type=int,
                        default=1000,
                        help="topk candidates for each mention in candidate generation phrase")
    parser.add_argument("--keep_N",
                        type=int,
                        default=1000,
                        help="keep keep_N candidates for each mention in shortlist phrase")
    parser.add_argument("--index_name",
                        type=str,
                        default="wikidata_20210830_keep_disambiguation")  # choices: wikidata_keep_disambiguation or wikidata_rm_disambiguation
    parser.add_argument("--in_links_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/wikidata-20210830/incoming_links/in_coming_links_num.pkl")
    parser.add_argument("--type_count_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/wikidata-20210830/type_count.pkl")
    parser.add_argument("--alias_map_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/wikidata-20210830/merged_alias_map/alias_map_keep_disambiguation.pkl") # choices: alias_map_keep_disambiguation.pkl alias_map_rm_disambiguation.pkl
    parser.add_argument("--id_mapping_fn",
                        type=str,
                        default="$BASE_DATA_DIR/mapping/wikidata_to_satori.json")
    parser.add_argument("--init_prune_topk", type=int, default=1000, help="candidate size after init")
    parser.add_argument("--alpha", type=float, default=0.20, help="weight for col_support_score")
    parser.add_argument("--beta", type=float, default=0.50, help="weight for row_support_score")
    parser.add_argument("--gamma", type=float, default=0.00, help="weight for popularity score: 1.0/rank")
    parser.add_argument("--max_iter", type=int, default=10, help="max iterations in ICA")
    parser.add_argument("--min_final_diff", type=float, default=0.00, help="use popularity when final score ties")
    parser.add_argument("--min_property_threshold", type=float, default=0.70)
    parser.add_argument("--use_slice", type=bool, default=True)
    parser.add_argument("--use_characteristics", type=bool, default=False)
    parser.add_argument("--row_feature_only", type=bool, default=False)
    parser.add_argument("--ent_feature", type=str, default="type_property")
    args = parser.parse_args()

    params = process_config(args)

    item_per_process = 3000
    max_row_size = 20
    number_process = 5
    if not os.path.exists(args.dump_dir):
        os.mkdir(args.dump_dir)
    parallel_processing_mini_tables(params,
                                    args.in_table_fn,
                                    args.dump_dir,
                                    item_per_process,
                                    max_row_size,
                                    number_process)

