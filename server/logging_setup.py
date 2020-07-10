from copy import deepcopy
from typing import Dict

import server.logging_utils


def get_logging_config() -> Dict:
    """
    Returns the logging config used by the duplicated code server
    """
    # Uses the default logging config set up in rdai.flask.logging_utils but changes the log level of the file handler
    # to INFO instead of DEBUG
    logging_config = deepcopy(logging_utils.LOGGING_CONFIG)
    logging_config['handlers']['file']['level'] = 'INFO'
    return logging_config


def setup_logging():
    if logging_utils.RUNNING_IN_GUNICORN:
        # Default full logging config applies to dev server and master process.
        # Workers logging will be configured by a post-fork hook in gunicorn
        # configuration.
        return
    logging_utils.setup_main_logging(logging_config=get_logging_config())
