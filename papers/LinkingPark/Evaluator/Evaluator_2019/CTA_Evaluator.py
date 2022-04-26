"""
The code is adapted from the official SemTab evaluator: https://github.com/sem-tab-challenge/aicrowd-evaluator
"""


import pandas as pd


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

    """
    # Evaluation for Round #1
    gt_num = 0
    gt_col_type = dict()
    gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'col_id', 'type'])
    for index, row in gt.iterrows():
        col = '%s %s' % (row['tab_id'], row['col_id'])
        gt_col_type[col] = row['type']
        gt_num += 1

    sub_num, correct_num = 0, 0
    sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'col_id', 'type'],
                      dtype={'tab_id': str, 'col_id': str, 'type': str})
    for index, row in sub.iterrows():
        sub_num += 1
        col = '%s %s' % (row['tab_id'], row['col_id'])
        if col in gt_col_type and gt_col_type[col] == row['type']:
            correct_num += 1

    precision = float(correct_num) / float(sub_num) if sub_num > 0 else 0.0
    recall = float(correct_num) / gt_num
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    main_score = f1
    secondary_score = precision
    """

    # Evaluation for Round #2
    cols, col_perfects, col_okays = set(), dict(), dict()
    gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'col_id', 'perfect', 'okay'],
                     dtype={'tab_id': str, 'col_id': str, 'perfect': str, 'okay': str}, keep_default_na=False)
    for index, row in gt.iterrows():
        col = '%s %s' % (row['tab_id'], row['col_id'])
        perfect_types = row['perfect'].lower().split(' ')
        okay_types = row['okay'].lower().split(' ')
        col_perfects[col] = perfect_types
        col_okays[col] = okay_types
        cols.add(col)

    annotations_num, perfect_annotations_num, okay_annotations_num, wrong_annotations_num = 0, 0, 0, 0
    annotated_cols = set()
    sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'col_id', 'annotations'],
                      dtype={'tab_id': str, 'col_id': str, 'annotations': str}, keep_default_na=False)
    for index, row in sub.iterrows():
        col = '%s %s' % (row['tab_id'], row['col_id'])
        if col in annotated_cols:
            raise Exception("Duplicate columns in the submission file")
        else:
            annotated_cols.add(col)
        annotations = set(row['annotations'].lower().split())

        if col in cols:
            annotations_num += len(annotations)
            for annotation in annotations:
                if annotation in col_perfects[col]:
                    perfect_annotations_num += 1
                elif annotation in col_okays[col]:
                    okay_annotations_num += 1
                else:
                    wrong_annotations_num += 1

    main_score = float(perfect_annotations_num + 0.5*okay_annotations_num - wrong_annotations_num) / len(cols)
    secondary_score = float(perfect_annotations_num) / float(annotations_num) if annotations_num > 0 else 0


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
    answer_file_path = "data/CTA_Round4_gt.csv"
    _client_payload = {}
    _client_payload["submission_file_path"] = "data/CTA_Round4_sub.csv"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234
    
    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = CTA_Evaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
