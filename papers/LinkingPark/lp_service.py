# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, abort
from flask import make_response

from TableAnnotator.Config import config
from TableAnnotator.Config.config_utils import process_config
from TableAnnotator.Detect.table_annotator import LinkingPark
from TableAnnotator.Log.logger import Logger
from TableAnnotator.Util.utils import InputTable

app = Flask(__name__)


@app.after_request
def apply_caching(response):
    # Adding default CORS headers as required by Excel Add-In
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
    return response


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'Error': 'Not found'}), 404)


@app.route('/autodetect/api/v1.0', methods=['GET'])
def status():
    return jsonify('OK'), 200


@app.route('/autodetect/api/v1.0/detect', methods=['POST'])
def detect():

    if not request.json or 'table' not in request.json:
        abort(400)

    t1 = datetime.now()
    logger.log(f'Input: {json.dumps(request.json)}')

    table = request.json["table"]
    input_tab = InputTable(table, str(uuid.uuid4()))

    satori_flag = request.args.get('sid')
    map_flag = satori_flag is not None and satori_flag.lower() in ['true', '1', 't', 'y', 'yes']

    output_tab = detector.detect_single_table(input_tab,
                                              keep_N=args.keep_N,
                                              alpha=args.alpha,
                                              beta=args.beta,
                                              gamma=args.gamma,
                                              topk=args.topk,
                                              init_prune_topk=params["init_prune_topk"],
                                              max_iter=args.max_iter,
                                              min_final_diff=args.min_final_diff,
                                              row_feature_only=params["row_feature_only"],
                                              map_ids=map_flag)

    info = output_tab.gen_online()
    logger.log(f'Output: {json.dumps(output_tab.dump_one_tab())}')

    t2 = datetime.now()
    delta = (t2 - t1).total_seconds()
    logger.log(f'Time Consumed: {delta}s\n')
    print(f'Time Consumed: {delta}s\n')

    return jsonify(info), 201


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=1000,
                        help="topk candidates for each mention in candidate generation phrase")
    parser.add_argument("--keep_N", type=int, default=1000,
                        help="keep keep_N candidates for each mention in shortlist phrase")
    parser.add_argument("--index_name",
                        type=str,
                        default="wikidata_rm_disambiguation")
    parser.add_argument("--in_links_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/incoming_links/in_coming_links_num.pkl")
    parser.add_argument("--alias_map_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/merged_alias_map/alias_map_rm_disambiguation.pkl")
    parser.add_argument("--id_mapping_fn",
                        type=str,
                        default="$BASE_DATA_DIR/mapping/wikidata_to_satori.json")
    parser.add_argument("--init_prune_topk", type=int, default=1000, help="candidate size after init")
    parser.add_argument("--alpha", type=float, default=0.20, help="weight for col_support_score")
    parser.add_argument("--beta", type=float, default=0.50, help="weight for row_support_score")
    parser.add_argument("--gamma", type=float, default=0.00, help="weight for popularity score: 1.0/rank")
    parser.add_argument("--max_iter", type=int, default=10, help="max iterations in ICA")
    parser.add_argument("--min_final_diff", type=float, default=0.00, help="use popularity when final score ties")
    parser.add_argument("--use_characteristics", type=bool, default=False)
    parser.add_argument("--row_feature_only", type=bool, default=False)
    parser.add_argument("--ent_feature", type=str, default="type_property")
    parser.add_argument("--port", type=int, default=6009)
    args = parser.parse_args()

    params = process_config(args)

    # cert_fn = os.path.join(config.base_system_dir, 'certs/fullchain.pem')
    # key_fn = os.path.join(config.base_system_dir, 'certs/privkey.pem')

    detector = LinkingPark(params)
    logger = Logger(config.log_fn)
    app.run(debug=False,
            host='0.0.0.0',
            port=config.port,
            # ssl_context=(cert_fn, key_fn)
            )

    print(f'LP listening on {config.port}...')
