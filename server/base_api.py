import logging
from abc import ABC, abstractmethod
from flask import request, jsonify, Flask, Response, Request

from logging_utils import stats, store_data_for_error_logging, logtime


logger = logging.getLogger(__name__)


class BasePOSTAPIPipeline(ABC):

    """
    Base class that defines a skeleton that encapsulates the logic to handle a REST API exposed by a flask app
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded = False

    @abstractmethod
    def parse_request(self, request: Request):
        """
        Parses the received request to extract the data that is needed
        """
        pass

    @abstractmethod
    def build_reply(self, processed_data):
        """
        From the processed data defines the logic and builds the reply to give has a response
        """
        pass

    @abstractmethod
    def process_request(self, parsed_data):
        """
        Defines the logic to process the data parsed from the request
        """
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """
        Defines the API version
        """
        pass

    @stats.timer("handle_request")
    def handle_request(self) -> Response:
        """
        Processes the received request and returns a response
        """
        class_name = self.__class__.__name__
        with logtime("entire job", logger, level='info'):
            stats.incr("requests")
            stats.incr("requests." + class_name)
            with logtime("parse data", logger, level='info'):
                parsed_data = self.parse_request(request)
                store_data_for_error_logging("request.data", parsed_data)
            processed_data = self.process_request(parsed_data)
            response = jsonify(self.build_reply(processed_data))
            stats.incr("replies")
        return response

    def __call__(self, *args, **kwargs): self.handle_request(*args, **kwargs)

    def _url_rule_options(self, options: dict) -> dict:
        """
        Methods allowed to this API. By default only POST methods are allowed
        """
        return {'methods': ['POST'], **(options or {})}

    def register(self, app: Flask, rule: str, options: dict = None):
        app.add_url_rule(rule, endpoint=self.__class__.__name__, view_func=self.handle_request,
                         **self._url_rule_options(options))

    def preload(self):
        """
        Indicates that the API is fully loaded and ready to receive requests
        """
        self.loaded = True

