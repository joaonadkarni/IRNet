import base64
import logging
import logging.config
import json
import time
from contextlib import contextmanager
from os import getenv
from typing import Dict

from flask import has_request_context, request, g
from statsd import StatsClient

RUNNING_IN_GUNICORN = getenv("RUNNING_IN_GUNICORN")
RUNNING_IN_GUNICORN_WORKER = getenv("RUNNING_IN_GUNICORN")


class NotDetailedFilter(logging.Filter):
    def filter(self, record):
        return (not RUNNING_IN_GUNICORN) or record.name != 'server.detail'


# Default logging config used by setup_main_logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': 'logs/server.log',
            'when': 'midnight',
            'formatter': 'file_fmt'
        },
        'debug': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'logs/debug.log',
            'formatter': 'file_fmt'
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': 'logs/error.log',
            'when': 'midnight',
            'formatter': 'error_file_fmt'
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'console_fmt',
            'filters': ['not_detailed']
        }
    },
    'filters': {
        'not_detailed': {
            '()': NotDetailedFilter,
        }
    },
    'formatters': {
        'console_fmt': {
            'format': '%(asctime).19s %(levelname)s %(process)d %(message)s'
        },
        'file_fmt': {
            'format': '%(asctime)s %(levelname)-8s %(process)d %(name)s %(message)s'
        },
        'error_file_fmt': {
            'format': ('[%(asctime)s %(levelname)-8s %(process)d %(name)s] '
                       '%(url)s %(trace_id)s\n'
                       '%(request_data)s\n'
                       '%(message)s')
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG'
        },
        'debug': {
            'handlers': ['console', 'debug'],
            'level': 'DEBUG'
        }
    }
}

default_log_factory = logging.getLogRecordFactory()


def log_record_factory_with_request_info(*args, **kwargs):
    """A log record factory method that adds info about the current request"""
    record = default_log_factory(*args, **kwargs)
    if has_request_context():
        record.url = request.url
        record.trace_id = request.headers.get("x-amzn-trace-id", None)
        record.remote_addr = request.remote_addr
        # only bother serializing request info if we're reporting errors or if
        # if that was explicitly asked for by the logger
        if (record.levelno >= logging.ERROR or kwargs.get('store_full_request', False)) and 'error_extra' in g:
            try:
                error_extra = g.error_extra
                request_log_data = {
                    k: base64.b85encode(v).decode('ascii') if isinstance(v, bytes) else v
                    for k, v in error_extra.items()
                }
                request_log_data['headers'] = request.headers.to_wsgi_list()
                request_log_data['method'] = request.method
                request_log_data['url'] = request.url
                record.request_data = json.dumps(request_log_data)
            except Exception as e:
                record.request_data = f'[[unable to serialize {e}]]'
        else:
            record.request_data = None
    else:
        record.url = None
        record.trace_id = None
        record.remote_addr = None
        record.request_data = None

    return record


logging.setLogRecordFactory(log_record_factory_with_request_info)


def store_data_for_error_logging(key, value):
    """Store generic data in the application context to be dumped to logs in case of an error.

    See https://flask.palletsprojects.com/en/1.1.x/appcontext/#storing-data
    """
    if 'error_extra' not in g:
        g.error_extra = {}
    g.error_extra[key] = value


def mk_logfile_dirs():
    from pathlib import Path
    for handler in LOGGING_CONFIG['handlers'].values():
        fpath = handler.get('filename', '')
        if not fpath: continue
        p = Path(fpath)
        p.parent.mkdir(exist_ok=True, parents=True)


def setup_main_logging(logging_config: Dict = LOGGING_CONFIG):
    mk_logfile_dirs()
    logging.config.dictConfig(logging_config)


def setup_remote_logging(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)


def run_remote_logging_server(queue):
    setup_main_logging()
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('** FAILURE WHILE WRITING LOG **:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


@contextmanager
def logtime(msg: str, logger: logging.Logger, level: str = 'debug'):
    start = time.time()
    try:
        yield
    finally:
        delta = time.time() - start
        log_msg = f"[TIMER] {msg} took {delta}"
        getattr(logger, level)(log_msg)


stats = StatsClient(prefix="server")


class StatsdLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        if record.levelname == 'ERROR':
            stats.incr("error.general")


# TODO: these should be run in `setup_main_logging` only
stats_error_handler = StatsdLogHandler()
stats_error_handler.setLevel(logging.ERROR)
