from __future__ import absolute_import, division, print_function

import datetime
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PackagePathFilter(logging.Filter):
    def filter(self, record):
        """add relative path to record
        """
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True



class Logger(object):
    def __init__(self, logger_name='None'):
        self.logger = logging.getLogger(logger_name)
        logging.root.setLevel(logging.NOTSET)
        self.log_file_name = 'uniqsar_{0}.log'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        cwd_path = os.path.abspath(os.getcwd())
        self.log_path = os.path.join(cwd_path, "logs")

        os.makedirs(self.log_path, exist_ok=True)
        self.backup_count = 5

        self.console_output_level = 'INFO'
        self.file_output_level = 'INFO'
        self.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        #self.formatter = logging.Formatter("%(asctime)s | %(relativepath)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s", self.DATE_FORMAT)
        self.formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", self.DATE_FORMAT)

    def get_logger(self):
        if not self.logger.handlers:
            # Console handler for stdout
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatter)
            console_handler.setLevel(self.console_output_level)
            console_handler.addFilter(PackagePathFilter())
            self.logger.addHandler(console_handler)

            # File handler for logging
            file_handler = TimedRotatingFileHandler(filename=os.path.join(self.log_path, self.log_file_name), when='D',
                        interval=1, backupCount=self.backup_count, delay=True, encoding='utf-8')
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(self.file_output_level)
            self.logger.addHandler(file_handler)

            # Set custom exception handler
            sys.excepthook = self.handle_exception

        self.logger.log_file_path = os.path.join(self.log_path, self.log_file_name)
        return self.logger

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        self.logger.error("Uncaught exception: %s", exc_value)

logger = Logger('Uni-QSAR').get_logger()
logger.setLevel(logging.INFO)