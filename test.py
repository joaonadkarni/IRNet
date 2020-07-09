from server_utils import get_and_load_model, build_input, buil_model_prediction_lf, generate_query_from_prediction_lf
from src import args as arg


if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    args = arg_parser.parse_args("--dataset ./data/custom --glove_embed_path ./data/glove.42B.300d.txt --cuda --epoch 50 --loss_epoch_threshold 50 --sketch_loss_coefficie 1.0 --beam_size 5 --seed 90 --save ${save_name} --embed_size 300 --sentence_features --column_pointer --hidden_size 300 --lr_scheduler --lr_scheduler_gammar 0.5 --att_vec_size 300 --load_model ./saved_model/IRNet_pretrained.model".split())

    model = get_and_load_model(args)

    while True:
        question_str = input("Enter your question: ")
        question_data, table_data_new = build_input(question_str)
        buil_model_prediction_lf(model, table_data_new, question_data, args.beam_size)
        generate_query_from_prediction_lf()
