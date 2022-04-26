"""
The code is adapted from the official SemTab evaluator: https://github.com/sem-tab-challenge/aicrowd-evaluator
"""

import pandas as pd
import os
import argparse
from prettytable import PrettyTable
from GlobalConfig import global_config


class CEA_Evaluator:
    def __init__(self, answer_file_path, round=1):
        """
    `round` : Holds the round for which the evaluation is being done. 
    can be 1, 2...upto the number of rounds the challenge has.
    Different rounds will mostly have different ground truth files.
    """
        self.answer_file_path = answer_file_path
        self.round = round

    def _evaluate(self, client_payload, _context={}):
        """
    `client_payload` will be a dict with (atleast) the following keys :
      - submission_file_path : local file path of the submitted file
      - aicrowd_submission_id : A unique id representing the submission
      - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
    """
        submission_file_path = client_payload["submission_file_path"]
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]

        gt_cell_ent = dict()
        gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'col_id', 'row_id', 'entity'],
                         dtype={'tab_id': str, 'col_id': str, 'row_id': str, 'entity': str}, keep_default_na=False)
        for index, row in gt.iterrows():
            cell = '%s %s %s' % (row['tab_id'], row['col_id'], row['row_id'])
            gt_cell_ent[cell] = row['entity']

        correct_cells, annotated_cells = set(), set()
        sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'col_id', 'row_id', 'entity'],
                          dtype={'tab_id': str, 'col_id': str, 'row_id': str, 'entity': str}, keep_default_na=False)
        for index, row in sub.iterrows():
            cell = '%s %s %s' % (row['tab_id'], row['col_id'], row['row_id'])
            if cell in gt_cell_ent:
                if cell in annotated_cells:
                    raise Exception("Duplicate cells in the submission file")
                else:
                    annotated_cells.add(cell)

                annotation = row['entity']
                if not annotation.startswith('http://www.wikidata.org/entity/'):
                    annotation = 'http://www.wikidata.org/entity/' + annotation
                if annotation.lower() == gt_cell_ent[cell].lower():
                    correct_cells.add(cell)

        precision = float(len(correct_cells)) / len(annotated_cells) if len(annotated_cells) > 0 else 0.0
        recall = float(len(correct_cells)) / len(gt_cell_ent.keys())
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        main_score = f1
        secondary_score = precision
        print('%.3f %.3f %.3f' % (f1, precision, recall))

        """
    Do something with your submitted file to come up
    with a score and a secondary score.

    if you want to report back an error to the user,
    then you can simply do :
      `raise Exception("YOUR-CUSTOM-ERROR")`

     You are encouraged to add as many validations as possible
     to provide meaningful feedback to your users
    """
        _result_object = {
            "score": main_score,
            "score_secondary": secondary_score
        }
        return _result_object


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=4)
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    if args.split == "all":
        answer_file_path = os.path.join(global_config.semtab_benchmark_dir,
                                        "GT/CEA/CEA_Round{}_gt.csv".format(args.round))
    else:
        answer_file_path = os.path.join(global_config.semtab_benchmark_dir,
                                        "GT/CEA/dev/CEA_Round{}_dev_gt.csv".format(args.round))

    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv

    results = PrettyTable()
    results.field_names = [
        "submission_name",
        "score",
        "score_secondary"
    ]

    for ff in os.listdir(args.data_dir):
        _client_payload = {}
        # print(ff)
        _client_payload["submission_file_path"] = os.path.join(args.data_dir, ff)
        _client_payload["aicrowd_submission_id"] = 1123
        _client_payload["aicrowd_participant_id"] = 1234

        # Instaiate a dummy context
        _context = {}
        # Instantiate an evaluator
        aicrowd_evaluator = CEA_Evaluator(answer_file_path)
        # Evaluate
        result = aicrowd_evaluator._evaluate(_client_payload, _context)
        # print(result)
        results.add_row([ff,
                         "{:.2f}".format(result['score'] * 100),
                         "{:.2f}".format(result['score_secondary'] * 100)])
    print(results)