import logging
from datetime import datetime
from flask import Request, Response

from base_api import BasePOSTAPIPipeline

from namedtuples import ServerModelInput, ServerResponse
from utils import get_and_load_model, get_args, build_model_prediction_lf, generate_query_from_prediction_lf, \
    build_spider_tables, build_input

logger = logging.getLogger(__name__)


class Nlp2SqlApiV1(BasePOSTAPIPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model, self.args = None, None

    def parse_request(self, request: Request) -> ModelInput:
        data = request.get_json()
        build_spider_tables(data['data_model'])
        question_data, table_data = build_input(data['text'])
        return ModelInput(question=question_data, tables=table_data)

    def handle_request(self, *args, **kwargs) -> Response:
        # Save timestamp when job was received
        job_date = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        logger.info(f"Received job at {job_date}")
        return super().handle_request(*args, **kwargs)

    def build_reply(self, processed_data: ServerResponse):
        return {
            "query_outputs": processed_data.out_ents,
            "query_sql": processed_data.sql,
        }

    def process_request(self, parsed_data: ModelInput) -> ServerResponse:
        """
        Defines the logic to process the data parsed from the request
        """
        build_model_prediction_lf(self.model, parsed_data.tables, parsed_data.question, self.args.beam_size)
        sql = generate_query_from_prediction_lf(logger)
        return ServerResponse(sql=sql, out_ents=["aa", "bb"])

    @property
    def version(self) -> str:
        return "1.0.0"

    def preload(self):
        self.args = get_args()
        self.model = get_and_load_model(self.args, logger)
        super().preload()
