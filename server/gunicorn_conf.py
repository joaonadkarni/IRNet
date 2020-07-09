########################
# gunicorn configuration

bind = '127.0.0.1:5000'
backlog = 2048

workers = 1
soft_start = True
worker_class = 'sync'
timeout = 60
# sync workers don't do keepalive
# keepalive = 1

proc_name = 'nlp2sql-server'
pidfile = '/var/run/ai/nlp2sql.pid'
errorlog = 'logs/gunicorn.log'
daemon = True


#######################
# Hooks and other logic

import logging
import os
import multiprocessing

from architecture_dashboard.module_classification.server.constants import SOFT_START_ENVVAR

os.environ['RUNNING_IN_GUNICORN'] = "1"

# remove automatic "optimizations" from anaconda that causes us
# to be bound to a single CPU on all workers
for k in ['KMP_AFFINITY', 'KMP_BLOCKTIME', 'KMP_SETTINGS']:
    if k in os.environ:
        del os.environ[k]

# this import needs to be after setting the above env variable
from rdai.flask.logging_utils import run_remote_logging_server, setup_remote_logging

# create an interprocess queue for log messages and
# spawn a process dedicated to logging
log_queue = multiprocessing.Queue(-1)
log_process = multiprocessing.Process(target=run_remote_logging_server, args=(log_queue,))
log_process.start()


if soft_start:
    expected_workers = workers
    workers = 1


def on_starting(server):
    """Called before the master is initialized

    Join the master process' own logs with the logserver.
    """
    setup_remote_logging(log_queue)
    # logging.getLogger("gunicorn.access").propagate = True
    # logging.getLogger("gunicorn.error").propagate = True


def when_ready(server):
    # logging.getLogger("gunicorn.access").propagate = True
    # logging.getLogger("gunicorn.error").propagate = True
    pass


def post_fork(server, worker):
    """Handler for worker right after coming alive.

    This hook runs in the worker process.
    Set the environment variable to signal logging
    """
    os.environ['RUNNING_IN_GUNICORN_WORKER'] = "1"


def pre_fork(server, worker):
    """Handler for master before spawning a new worker

    This hook runs in the master process.
    """
    # if we're going through the soft-start cycle, we signal each worker to
    # ask the master to spawn the next worker
    os.environ['GUNICORN_INIT_NOTIFY_PARENT'] = "1" if soft_start else "0"
    pass


def post_worker_init(worker):
    """Called just after a worker has initialized the application"""
    if os.environ['GUNICORN_INIT_NOTIFY_PARENT'] == "1":
        import signal
        os.kill(worker.ppid, signal.SIGTTIN)


def nworkers_changed(server, new_value, old_value):
    global soft_start
    if soft_start:
        if new_value == expected_workers:
            # if we reached the expected number of workers, the soft-start cycle
            # is over
            soft_start = False


def worker_int(worker):
    """Handler for worker getting INT or QUIT signals.

    This hook runs on the worker process.
    Besides the default behavior of restarting the worker,
    we'll also dump the stacktrace of all running threads.
    """
    worker.log.info("worker received INT or QUIT signal")

    ## get traceback info
    import threading, sys, traceback
    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""),
                                            threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename,
                                                        lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    worker.log.debug("\n".join(code))


def on_exit(server):
    server.log.info("exiting master process")

    # reset log handling sink for this process
    root = logging.getLogger()
    for h in root.handlers:
        root.removeHandler(h)
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    root.addHandler(h)

    # disconnect logger process
    log_queue.put_nowait(None)
    log_process.join()
