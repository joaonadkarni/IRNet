import logging
import argparse
import sys

import logging_setup
from api import Nlp2SqlApiV1
from base_app import BaseFlaskApp
from logging_utils import stats_error_handler


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def parse_arguments():
    parser = ArgParser(description="Launch the module classification server",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help="IP on which to run the module classification server.")
    parser.add_argument('--port', type=int, default=5000, help="Port on which to run the module classification server.")
    return parser.parse_args()


class NLP2SQLApp(BaseFlaskApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.known_apis.update({"/api/nlp2sql/predict/v1": Nlp2SqlApiV1()})
        self.production_apis = [
            "/api/nlp2sql/predict/v1"
        ]


def create_app():
    app = NLP2SQLApp(__name__)
    app.register_pipelines()
    app.logger.addHandler(stats_error_handler)
    logging_setup.setup_logging()
    app.preload()
    logger = logging.getLogger(__name__)
    logger.info(f'App ready')
    return app


if __name__ == '__main__':
    args = parse_arguments()
    app = create_app()
    app.run(threaded=False, host=args.host, port=args.port)
