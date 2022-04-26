"""
The code is adapted from the official SemTab evaluator: https://github.com/sem-tab-challenge/aicrowd-evaluator
"""

import json
import os

import pandas as pd
from GlobalConfig import global_config

try:
    from .TT import get_tables_categories
except ImportError as e:
    from TT import get_tables_categories

class CTA_Evaluator:
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

        gt_ancestor = json.load(open(os.path.join(global_config.tough_table_benchmark_dir, "gt/CTA_2T_WD_gt_ancestor.json")))
        gt_descendent = json.load(open(os.path.join(global_config.tough_table_benchmark_dir, "gt/CTA_2T_WD_gt_descendent.json")))

        gt_cols, col_type = set(), dict()
        gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'col_id', 'type', 'label'],
                         dtype={'tab_id': str, 'col_id': str, 'type': str, 'label': str}, keep_default_na=False)

        # FIX GT error: this column was reported with col_id = 1 in the original target file.
        gt = gt[~(gt['tab_id'].isin(['24W5SSRB', '3LG8J4MX']) & (gt['col_id'] == "2"))]

        for index, row in gt.iterrows():
            col = (row['tab_id'], row['col_id'])
            gt_cols.add(col)
            col_type[col] = row['type'].split()

        correct_cols, annotated_cols = dict(), set()
        sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'col_id', 'annotation'],
                          dtype={'tab_id': str, 'col_id': str, 'annotation': str}, keep_default_na=False)
        for index, row in sub.iterrows():
            col = (row['tab_id'], row['col_id'])
            if col in gt_cols:  # Ignore columns out of target
                if col in annotated_cols:
                    raise Exception("Duplicate columns in the submission file")
                else:
                    annotated_cols.add(col)
                annotation = row['annotation']
                if annotation and not annotation.startswith('http://www.wikidata.org/entity/'):
                    annotation = 'http://www.wikidata.org/entity/' + annotation

                gt_types = col_type[col]
                ancestor = dict()
                descendent = dict()
                for gt_type in gt_types:  # collect the ancestors/descendents of all the GT types
                    ancestor.update(gt_ancestor[gt_type])
                    descendent.update(gt_descendent[gt_type])
                ancestor = {a.lower(): b for a, b in ancestor.items()}
                descendent = {a.lower(): b for a, b in descendent.items()}

                if annotation.lower() in [x.lower() for x in gt_types]:
                    correct_cols[col] = 1.0
                elif annotation.lower() in ancestor:
                    depth = int(ancestor[annotation.lower()])
                    correct_cols[col] = pow(0.8, depth) if depth <= 5 else 0.0
                elif annotation.lower() in descendent:
                    depth = int(descendent[annotation.lower()])
                    correct_cols[col] = pow(0.7, depth) if depth <= 3 else 0.0
                else:
                    correct_cols[col] = 0

        main_score = .0
        secondary_score = .0

        # 2T detailed scores
        for cat, table_list in get_tables_categories().items():
            c_cols = {col: score for col, score in correct_cols.items() if col[0] in table_list}
            a_cols = {x for x in annotated_cols if x[0] in table_list}
            g_cols = {x for x in gt_cols if x[0] in table_list}

            if len(g_cols) > 0:
                total_score = sum(c_cols.values())
                precision = total_score / len(a_cols) if len(a_cols) > 0 else 0
                recall = total_score / len(g_cols)
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                print('%s %.3f %.3f %.3f' % (cat, f1, precision, recall))
                if cat == 'ALL':
                    main_score = f1
                    secondary_score = precision

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
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    answer_file_path = "DataSets/2T_WD/gt/CTA_2T_WD_gt.csv"

    d = 'SUB/CTA/'
    for ff in os.listdir(d):
        _client_payload = {}
        print(ff)
        _client_payload["submission_file_path"] = d + ff
        _client_payload["aicrowd_submission_id"] = 1123
        _client_payload["aicrowd_participant_id"] = 1234

        # Instaiate a dummy context
        _context = {}
        # Instantiate an evaluator
        aicrowd_evaluator = CTA_Evaluator(answer_file_path)
        # Evaluate
        result = aicrowd_evaluator._evaluate(_client_payload, _context)
        print(result)
