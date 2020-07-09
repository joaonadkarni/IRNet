from server.utils import get_and_load_model, build_input, build_model_prediction_lf, generate_query_from_prediction_lf, \
    get_args
from src import args as arg


if __name__ == '__main__':
    args = get_args()
    model = get_and_load_model(args)

    while True:
        question_str = input("Enter your question: ")
        question_data, table_data_new = build_input(question_str)
        build_model_prediction_lf(model, table_data_new, question_data, args.beam_size)
        generate_query_from_prediction_lf()
