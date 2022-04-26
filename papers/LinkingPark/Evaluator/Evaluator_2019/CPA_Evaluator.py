"""
The code is adapted from the official SemTab evaluator: https://github.com/sem-tab-challenge/aicrowd-evaluator
"""

import pandas as pd


class CPA_Evaluator:
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

    gt_cols_pro = dict()
    gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'sub_col_id', 'obj_row_id', 'property'],
                     dtype={'tab_id': str, 'sub_col_id': str, 'obj_row_id': str, 'property': str}, keep_default_na=False)
    for index, row in gt.iterrows():
        cols = '%s %s %s' % (row['tab_id'], row['sub_col_id'], row['obj_row_id'])
        gt_cols_pro[cols] = row['property'].lower().split()

    annotated_cols, correct_cols = set(), set()
    sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'sub_col_id', 'obj_row_id', 'property'],
                      dtype={'tab_id': str, 'sub_col_id': str, 'obj_row_id': str, 'property': str}, keep_default_na=False)
    for index, row in sub.iterrows():
        cols = '%s %s %s' % (row['tab_id'], row['sub_col_id'], row['obj_row_id'])
        if cols in gt_cols_pro:
            if cols in annotated_cols:
                raise Exception("Duplicate column pairs in the submission file")
            else:
                annotated_cols.add(cols)
            if row['property'].lower() in gt_cols_pro[cols]:
                correct_cols.add(cols)

    precision = float(len(correct_cols)) / len(annotated_cols) if len(annotated_cols) > 0 else 0.0
    recall = float(len(correct_cols)) / len(gt_cols_pro.keys())
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
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
    answer_file_path = "data/CPA_Round4_gt.csv"
    _client_payload = {}
    _client_payload["submission_file_path"] = "data/CPA_Round4_sub.csv"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234
    
    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = CPA_Evaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
