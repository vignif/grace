# -*-coding:utf-8-*-
import logging
import os
import sys
import logging.config

os.makedirs('logs', exist_ok=True)


class Logger:
    def __init__(self, name, level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        stream_handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "[%(levelname)s] [%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
            "%Y-%m-%d %H:%M:%S")
        stream_handler.setFormatter(formatter)

        stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)
        self.logger.propagate = False
