import json
import subprocess
import traceback

import torch

from constants import MODEL_PATH, EMBEDS_PATH, QUESTION_DATA_PATH, TABLE_DATA_PATH, PREDICT_LF_PATH
from dummy_input import DUMMY_INPUT
from sem2SQL import transform
from src.models.model import IRNet
from src.rule import semQL
from src.utils import load_data_new, get_json_data, load_word_emb
from src import args as arg

from src.rule.sem_utils import alter_column0, alter_inter, alter_not_in


def _log_or_print(logger, msg):
    if logger:
        logger.info(msg)
    else:
        print(msg)


def get_args():
    arg_parser = arg.init_arg_parser()
    return arg_parser.parse_args(
        "--dataset ./data/custom --glove_embed_path ./data/glove.42B.300d.txt --cuda --epoch 50 "
        "--loss_epoch_threshold 50 --sketch_loss_coefficie 1.0 --beam_size 5 --seed 90 --save ${save_name} "
        "--embed_size 300 --sentence_features --column_pointer --hidden_size 300 --lr_scheduler --lr_scheduler_gammar "
        "0.5 --att_vec_size 300 --load_model ./saved_model/IRNet_pretrained.model".split())


def get_and_load_model(args, logger=None):
    grammar = semQL.Grammar()
    model = IRNet(args, grammar)

    if args.cuda: model.cuda()

    _log_or_print(logger, f"Loading pretrained model from {MODEL_PATH}")

    pretrained_model = torch.load(MODEL_PATH,
                                  map_location=lambda storage, loc: storage)
    import copy
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    model.word_emb = load_word_emb(EMBEDS_PATH)

    model.eval()

    return model


def build_input(question_str):

    DUMMY_INPUT["question"] = question_str
    DUMMY_INPUT["question_toks"] = [tok for tok in question_str.strip().split(" ") if tok]

    if DUMMY_INPUT["question_toks"][-1].endswith("?"):
        DUMMY_INPUT["question_toks"][-1] = DUMMY_INPUT["question_toks"][-1][:-1]
        DUMMY_INPUT["question_toks"].append("?")

    import json
    with open(QUESTION_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump([DUMMY_INPUT], f, ensure_ascii=False, indent=4)

    subprocess.run(["cd preprocess && bash run_me.sh ../data/custom/question.json ../data/custom/tables.json"],
                   shell=True, stdout=subprocess.DEVNULL)

    with open(TABLE_DATA_PATH) as inf:
        # print("Loading data from %s" % TABLE_DATA_PATH)
        table_data = json.load(inf)

    question_data, table_data_new = load_data_new(QUESTION_DATA_PATH, table_data, use_small=False)

    return question_data, table_data_new


def build_model_prediction_lf(model, table_data_new, question_data, beam_size):
    json_datas = get_json_data(model, table_data_new, question_data, beam_size=beam_size)
    with open(PREDICT_LF_PATH, 'w') as f:
        json.dump(json_datas, f)


def generate_query_from_prediction_lf(logger=None):

    def _load_dataSets(input_path, tables_path):
        with open(input_path, 'r') as f:
            datas = json.load(f)
        with open(tables_path, 'r', encoding='utf8') as f:
            table_datas = json.load(f)
        schemas = dict()
        for i in range(len(table_datas)):
            schemas[table_datas[i]['db_id']] = table_datas[i]
        return datas, schemas

    datas, schemas = _load_dataSets(input_path=PREDICT_LF_PATH, tables_path=TABLE_DATA_PATH)
    assert len(datas) == 1, "More than 1 output query"
    alter_not_in(datas, schemas=schemas)
    alter_inter(datas)
    alter_column0(datas)

    data = datas[0]

    try:
        result = transform(data, schemas[data['db_id']])
        _log_or_print(logger, f"Query: {result[0]}")
    except Exception as e:
        result = transform(data, schemas[data['db_id']],
                           origin='Root1(3) Root(5) Sel(0) N(0) A(3) C(0) T(0)')
        _log_or_print(logger, f"Query: {result[0]}")
        if logger:
            logger.error("Something went wrong transforming the predicted lf to the sql query", exc_info=1)
        else:
            print(e)
            print('Exception')
            print(traceback.format_exc())
            print('===\n\n')

    return result[0]


