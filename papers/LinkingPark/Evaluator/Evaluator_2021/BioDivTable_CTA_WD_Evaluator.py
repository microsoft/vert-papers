"""
The code is adapted from the official SemTab evaluator: https://github.com/sem-tab-challenge/aicrowd-evaluator
"""

import pandas as pd
import json
import os
import argparse


class CTA_Evaluator:
    def __init__(self, answer_file_path, round=1):
        """
        `round` : Holds the round for which the evaluation is being done.
        can be 1, 2...upto the number of rounds the challenge has.
        Different rounds will mostly have different ground truth files.
        """
        self.answer_file_path = answer_file_path
        self.round = round

    def _evaluate(self, client_payload, gt_ancestor_fn, gt_descendent_fn, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]

        gt_ancestor = json.load(open(gt_ancestor_fn))
        gt_descendent = json.load(open(gt_descendent_fn))

        cols, col_type = set(), dict()
        gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'col_id', 'type'],
                         dtype={'tab_id': str, 'col_id': str, 'type': str}, keep_default_na=False)
        for index, row in gt.iterrows():
            col = '%s %s' % (row['tab_id'], row['col_id'])
            gt_type = row['type']
            col_type[col] = gt_type
            cols.add(col)

        annotated_cols = set()
        total_score = 0
        sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'col_id', 'annotation'],
                          dtype={'tab_id': str, 'col_id': str, 'annotation': str}, keep_default_na=False)
        for index, row in sub.iterrows():
            col = '%s %s' % (row['tab_id'], row['col_id'])
            if col in annotated_cols:
                # continue
                raise Exception("Duplicate columns in the submission file")
            else:
                annotated_cols.add(col)
            annotation = row['annotation']
            # if not annotation.startswith('http://www.wikidata.org/entity/'):
            #     annotation = 'http://www.wikidata.org/entity/' + annotation

            if col in cols:
                max_score = 0
                for gt_type in col_type[col].strip().split(','):
                    gt_type = gt_type.strip()
                    if gt_type not in gt_ancestor:
                        print(gt_type)
                    ancestor = gt_ancestor.get(gt_type, {})
                    ancestor_keys = [k.lower() for k in ancestor]
                    # descendent = gt_descendent[gt_type]
                    # https://github.com/fusion-jena/BiodivTab/issues/1
                    descendent = gt_descendent.get(gt_type, {})
                    descendent_keys = [k.lower() for k in descendent]
                    if annotation.lower() == gt_type.lower():
                        score = 1.0
                    elif annotation.lower() in ancestor_keys:
                        depth = int(ancestor[annotation])
                        if depth <= 5:
                            score = pow(0.8, depth)
                        else:
                            score = 0
                    elif annotation.lower() in descendent_keys:
                        depth = int(descendent[annotation])
                        if depth <= 3:
                            score = pow(0.7, depth)
                        else:
                            score = 0
                    else:
                        score = 0
                    if score > max_score:
                        max_score = score

                total_score += max_score

        precision = total_score / len(annotated_cols) if len(annotated_cols) > 0 else 0
        recall = total_score / len(cols)
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
    parser.add_argument("--gt-dir",
                        type=str,
                        required=True)
    parser.add_argument("--submission-fn",
                        type=str,
                        required=True)
    args = parser.parse_args()
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    answer_file_path = os.path.join(args.gt_dir, "CTA_biodivtab_2021_gt.csv")
    _client_payload = {}
    _client_payload["submission_file_path"] = args.submission_fn
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = CTA_Evaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload,
                                         os.path.join(args.gt_dir, "CTA_biodivtab_2021_WD_gt_ancestor.json"),
                                         os.path.join(args.gt_dir, "CTA_biodivtab_2021_WD_gt_descendent.json"),
                                         _context)
    print(result)