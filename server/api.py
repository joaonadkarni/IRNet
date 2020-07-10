import logging
from datetime import datetime
from flask import Request, Response

from server.base_api import BasePOSTAPIPipeline

from server.namedtuples import ServerModelInput, ServerResponse
from server.utils import get_and_load_model, get_args, build_model_prediction_lf, \
    generate_query_and_out_attrs_from_prediction_lf, build_spider_tables, build_input

logger = logging.getLogger(__name__)


class Nlp2SqlApiV1(BasePOSTAPIPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model, self.args = None, None

    def parse_request(self, request: Request) -> ServerModelInput:
        data = request.get_json()
        add_all_type_str = request.args.get('alltype', "0")
        add_all_type = True if add_all_type_str == "1" else False
        db_id = build_spider_tables(data['data_model'], add_all_type=add_all_type)
        question_data, table_data = build_input(data['text'], db_id=db_id)
        return ServerModelInput(question=question_data, tables=table_data)

    def handle_request(self, *args, **kwargs) -> Response:
        # Save timestamp when job was received
        job_date = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        logger.info(f"Received job at {job_date}")
        return super().handle_request(*args, **kwargs)

    def build_reply(self, processed_data: ServerResponse):
        return {
            "query_outputs": processed_data.out_attrs,
            "query_sql": processed_data.sql,
            "query_inputs": processed_data.input_params
        }

    def process_request(self, parsed_data: ServerModelInput) -> ServerResponse:
        """
        Defines the logic to process the data parsed from the request
        """
        build_model_prediction_lf(self.model, parsed_data.tables, parsed_data.question, self.args.beam_size)
        sql, out_attrs, input_params = generate_query_and_out_attrs_from_prediction_lf(logger)
        return ServerResponse(sql=sql, out_attrs=out_attrs, input_params=input_params)

    @property
    def version(self) -> str:
        return "1.0.0"

    def preload(self):
        self.args = get_args()
        self.model = get_and_load_model(self.args, logger)
        super().preload()
