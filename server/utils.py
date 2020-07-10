import json
import subprocess
import traceback
import pandas as pd

import torch

from server.constants import MODEL_PATH, EMBEDS_PATH, QUESTION_DATA_PATH, TABLE_DATA_PATH, PREDICT_LF_PATH
from server.dummy_input import DUMMY_INPUT
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


def _normalize_word(word):
    if len(word) < 2:
        return [word.lower()]

    tokens = []
    b = 0 if (word[0].isalpha()) else 1
    i = 1

    for char in word[1:]:

        # If the word has a lower case letter followed by an upper case letter then it's separated
        #  in two subtokens, with the uppercase letter being the cutting point
        # For example, createdBy would be divided in created and by
        # Note that the uppercase letter of the second word is replaced by its correspondent lower case letter
        if char.isupper() and word[i - 1].islower() and (i - b) > 1:
            tokens.append(word[b:i].lower())
            b = i

        # If the word has a sequence of upper case letters followed by a lower case letter then it's separated
        #  in two subtokens, with the last upper letter being the cutting point
        # For example, FRAPScase would be divided in frap and scase
        # Note that the uppercase letters are replaced by its correspondent lower case letters
        elif char.islower() and word[i - 1].isupper() and (i - b) > 1:
            if (i - b > 2):
                tokens.append(word[b:i - 1].lower())
            b = i - 1

        # If the word has an alpha numeric character followed by a non alpha numeric character then it's separated
        #  in two subtokens, with the non alphabetic character being the cutting point
        # For example, created_By would be divided in created and by
        # Note that the uppercase letter of the second word is replaced by its correspondent lower case letter
        # Note2: the word has to have at least 2 characters to be considered valid
        elif (not char.isalpha()):
            if (word[i - 1].isalpha()) and (i - b) > 1:
                tokens.append(word[b:i].lower())
            b = i + 1

        i += 1

    # Add last sub token
    # Note that the uppercase letter of the second word is replaced by its correspondent lower case letter
    # Note2: the word has to have at least 2 characters to be considered valid
    if b < len(word) and (i - b) > 1:
        tokens.append(word[b:i].lower())

    return " ".join(tokens)


def _clean_entities(entities):
    return [ent for ent in entities if not (ent['isHidden'] or ent['isStatic'] or (ent['refKey'] is not None))]


def _get_data_model_df(data_model):
    entities = data_model['entities']
    augmented_attributes = []
    # for now we just consider locally created entities
    local_entities = _clean_entities(entities)
    for idx, ent in enumerate(local_entities):
        for attr in ent['attributes']:
            attr['entIdx'] = idx
            attr['entKey'] = ent['key']
            attr['entName'] = ent['name']
            attr['entIsHidden'] = ent['isHidden']
            attr['entIsStatic'] = ent['isStatic']
            attr['entRefKey'] = ent['refKey']
            augmented_attributes.append(attr)
    return pd.DataFrame(augmented_attributes)


def _data_model_to_spider_data(data_model):
    df = _get_data_model_df(data_model)
    map_types = dict({'Text': 'text',
                      'Integer': 'number',
                      'Decimal': 'number',
                      'Long Integer': 'number',
                      'Boolean': 'number',
                      'Date time': 'time',
                      'Time': 'time',
                      'Date': 'time',
                      'Phone Number': 'number',
                      'Email': 'text',
                      'Currency': 'number',
                      'Categorical': 'text',
                      'Identifier': 'number'})

    df['index'] = df.index + 1
    all_column = [-1, "*"]

    result = {
        'column_names': [all_column, *df.apply(lambda x: [x['entIdx'], _normalize_word(x['name'])], axis=1).tolist()],
        'column_names_original': [all_column, *df.apply(lambda x: [x['entIdx'], x['name']], axis=1).tolist()],
        'column_types': df['type'].map(
            lambda x: 'number' if x.endswith('Identifier') else map_types.get(x, 'text')).tolist(),
        'db_id': '_'.join(df['entName'].unique().tolist()),
        'primary_keys': df.loc[df['isPrimary'] == True]['index'].tolist(),
        'table_names': [_normalize_word(x) for x in df['entName'].unique().tolist()],
        'table_names_original': df['entName'].unique().tolist()
    }

    fks = []
    for fkref, idx in df.loc[~df['fkRefKey'].isnull()][['fkRefKey', 'index']].values.tolist():
        origin_key_idx = df.loc[(df['entKey'] == fkref) & (df['isPrimary'] == True)]['index'].tolist()
        if len(origin_key_idx) == 1:
            fks.append([idx, origin_key_idx[0]])

    result['foreign_keys'] = fks

    return result


def build_spider_tables(data_model):
    spider_table = _data_model_to_spider_data(data_model)
    with open(TABLE_DATA_PATH, 'w') as f:
        json.dump([spider_table], f)
    return spider_table['db_id']


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


def build_input(question_str, db_id=None):
    DUMMY_INPUT["question"] = question_str
    DUMMY_INPUT["question_toks"] = [tok for tok in question_str.strip().split(" ") if tok]
    if db_id:
        DUMMY_INPUT["db_id"] = db_id

    if len(DUMMY_INPUT["question_toks"][-1]) > 1 and DUMMY_INPUT["question_toks"][-1].endswith("?"):
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
