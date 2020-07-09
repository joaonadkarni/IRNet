import traceback

import torch

from sem2SQL import transform
from src import args as arg
from src import utils
from src.models.model import IRNet
from src.rule import semQL

import subprocess

from dummy_input import DUMMY_INPUT
from src.rule.sem_utils import alter_column0, alter_inter, alter_not_in
from src.utils import load_data_new

MODEL_PATH = "./saved_model/IRNet_pretrained.model"
EMBEDS_PATH = "./data/glove.42B.300d.txt"
QUESTION_DATA_PATH = "./data/custom/question.json"
TABLE_DATA_PATH = "./data/custom/tables.json"
PREDICT_LF_PATH = "./data/custom/predict_lf.json"


def evaluate(args):
    """
    :param args:
    :return:
    """

    grammar = semQL.Grammar()
    model = IRNet(args, grammar)

    if args.cuda: model.cuda()

    print('load pretrained model from %s' % MODEL_PATH)
    pretrained_model = torch.load(MODEL_PATH,
                                  map_location=lambda storage, loc: storage)
    import copy
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    model.word_emb = utils.load_word_emb(EMBEDS_PATH)

    model.eval()

    while True:

        question_str = input("Enter your question: ")

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
            print("Loading data from %s" % TABLE_DATA_PATH)
            table_data = json.load(inf)

        question_data, table_data_new = load_data_new(QUESTION_DATA_PATH, table_data, use_small=False)

        json_datas = utils.get_json_data(model, table_data_new, question_data, beam_size=args.beam_size)
        with open(PREDICT_LF_PATH, 'w') as f:
            json.dump(json_datas, f)

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
            print(f"Query: {result[0]}")
        except Exception as e:
            result = transform(data, schemas[data['db_id']],
                               origin='Root1(3) Root(5) Sel(0) N(0) A(3) C(0) T(0)')
            print(f"Query: {result[0]}")
            print(e)
            print('Exception')
            print(traceback.format_exc())
            print('===\n\n')


if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    args = arg_parser.parse_args("--dataset ./data/custom --glove_embed_path ./data/glove.42B.300d.txt --cuda --epoch 50 --loss_epoch_threshold 50 --sketch_loss_coefficie 1.0 --beam_size 5 --seed 90 --save ${save_name} --embed_size 300 --sentence_features --column_pointer --hidden_size 300 --lr_scheduler --lr_scheduler_gammar 0.5 --att_vec_size 300 --load_model ./saved_model/IRNet_pretrained.model".split())
    evaluate(args)
