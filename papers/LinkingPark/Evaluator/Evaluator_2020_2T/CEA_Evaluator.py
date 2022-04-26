"""
The code is adapted from the official SemTab evaluator: https://github.com/sem-tab-challenge/aicrowd-evaluator
"""

import os

import pandas as pd

try:
    from .TT import get_tables_categories
except ImportError as e:
    from TT import get_tables_categories


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
        # The SemTab2020 CEA submission format is tab_id, row_id, col_id, entity
        gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'row_id', 'col_id', 'entity'],
                         dtype={'tab_id': str, 'row_id': str, 'col_id': str, 'entity': str}, keep_default_na=False)

        # FIX GT error: this column was reported with col_id = 1 in the original target file.
        gt = gt[~(gt['tab_id'].isin(['24W5SSRB', '3LG8J4MX']) & (gt['col_id'] == "2"))]

        for index, row in gt.iterrows():
            cell = (row['tab_id'], row['row_id'], row['col_id'])
            gt_cell_ent[cell] = row['entity'].lower().split()

        correct_cells, annotated_cells = set(), set()
        sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'row_id', 'col_id', 'entity'],
                          dtype={'tab_id': str, 'row_id': str, 'col_id': str, 'entity': str}, keep_default_na=False)

        for index, row in sub.iterrows():
            cell = (row['tab_id'], row['row_id'], row['col_id'])
            if cell in gt_cell_ent:  # Ignore cells out of target
                if cell in annotated_cells:
                    raise Exception("Duplicate cells in the submission file")
                else:
                    annotated_cells.add(cell)

                annotation = row['entity']
                if not annotation:
                    if gt_cell_ent[cell] == 'nil':  # Knowledge gap
                        correct_cells.add(cell)
                else:
                    if not annotation.startswith('http://www.wikidata.org/entity/'):
                        annotation = 'http://www.wikidata.org/entity/' + annotation
                    if annotation.lower() in gt_cell_ent[cell]:
                        correct_cells.add(cell)

        main_score = .0
        secondary_score = .0

        # 2T detailed scores
        for cat, table_list in get_tables_categories().items():
            c_cells = {x for x in correct_cells if x[0] in table_list}
            a_cells = {x for x in annotated_cells if x[0] in table_list}
            g_cells = {x for x in gt_cell_ent if x[0] in table_list}

            if len(g_cells) > 0:
                precision = float(len(c_cells)) / len(a_cells) if len(a_cells) > 0 else 0.0
                recall = float(len(c_cells)) / len(g_cells)
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
    answer_file_path = "DataSets/2T_WD/gt/CEA_2T_WD_gt.csv"

    d = 'SUB/CEA/'
    for ff in os.listdir(d):
        _client_payload = {}
        print(ff)
        _client_payload["submission_file_path"] = d + ff
        _client_payload["aicrowd_submission_id"] = 1123
        _client_payload["aicrowd_participant_id"] = 1234

        # Instaiate a dummy context
        _context = {}
        # Instantiate an evaluator
        aicrowd_evaluator = CEA_Evaluator(answer_file_path)
        # Evaluate
        result = aicrowd_evaluator._evaluate(_client_payload, _context)
        print(result)
