import pandas as pd
import os
import argparse

prefix1 = 'http://www.wikidata.org/entity/'
prefix2 = 'https://www.wikidata.org/wiki/'


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
        gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'row_id', 'col_id', 'entity'],
                         dtype={'tab_id': str, 'row_id': str, 'col_id': str, 'entity': str}, keep_default_na=False)
        for index, row in gt.iterrows():
            cell = '%s %s %s' % (row['tab_id'], row['row_id'], row['col_id'])

            """
                Trim the entity prefix to avoid mismatching
            """

            row['entity'] = row['entity'].replace(prefix1, '')
            row['entity'] = row['entity'].replace(prefix2, '')

            gt_cell_ent[cell] = row['entity']

        correct_cells, annotated_cells = set(), set()
        sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'row_id', 'col_id', 'entity'],
                          dtype={'tab_id': str, 'row_id': str, 'col_id': str, 'entity': str}, keep_default_na=False)
        for index, row in sub.iterrows():
            cell = '%s %s %s' % (row['tab_id'], row['row_id'], row['col_id'])
            if cell in gt_cell_ent:
                if cell in annotated_cells:
                    raise Exception("Duplicate cells in the submission file")
                else:
                    annotated_cells.add(cell)

                annotation = row['entity']
                # if not annotation.startswith('http://www.wikidata.org/entity/'):
                #     annotation = 'http://www.wikidata.org/entity/' + annotation

                """
                    Trim the entity prefix to avoid mismatching 
                """
                if annotation.startswith(prefix1):
                    annotation = annotation[len(prefix1):]

                if annotation.startswith(prefix2):
                    annotation = annotation[len(prefix2):]

                if annotation.lower() in gt_cell_ent[cell].lower().split():
                    correct_cells.add(cell)

        precision = len(correct_cells) / len(annotated_cells) if len(annotated_cells) > 0 else 0.0
        recall = len(correct_cells) / len(gt_cell_ent.keys())
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
    parser.add_argument("--gt-fn",
                        type=str,
                        required=True)
    parser.add_argument("--submission-fn",
                        type=str,
                        required=True)
    args = parser.parse_args()
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    answer_file_path = args.gt_fn


    _client_payload = {}
    _client_payload["submission_file_path"] = args.submission_fn
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = CEA_Evaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
