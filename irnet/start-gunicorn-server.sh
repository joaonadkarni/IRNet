#!/bin/sh
GUNICORN_CONF=server/gunicorn_conf.py
if ! [ -r "$GUNICORN_CONF" ]; then
    echo "Run this script from the root of the repository" >&2
    echo "(failed to start because couldn't find ${GUNICORN_CONF})" >&2
    exit 1
fi
gunicorn "main:create_app()" $*
tail -F logs/gunicorn.log