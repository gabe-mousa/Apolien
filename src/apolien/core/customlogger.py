import textwrap
import logging
import sys
from . import testsettings
import pprint
import os


class CustomFormatter(logging.Formatter):
    def __init__(self, width=80):
        super().__init__()
        self.width = width
    
    def format(self, record):
        # If the message is a string, wrap it
        if isinstance(record.msg, str):
                # Split the text into paragraphs by newlines
                paragraphs = record.msg.split('\n')
                # Wrap each paragraph separately
                wrapped_paragraphs = [
                    textwrap.fill(p, width=self.width) if p.strip() else ''
                    for p in paragraphs
                ]
                # Rejoin with original newlines
                record.msg = '\n'.join(wrapped_paragraphs)
        else:
            # For non-string objects, use pretty print
            record.msg = pprint.pformat(record.msg, sort_dicts=False)
        return record.msg


def setupLogger(toFile, filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    filename = testsettings.testResultsDir + "/" + filename
    
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    if toFile:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.isfile(filename):
            os.remove(filename)
        with open(filename, 'x') as _:
            pass
        handler = logging.FileHandler(filename)
    else:
        handler = logging.StreamHandler(sys.stdout)

    formatter = CustomFormatter(width=80)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def setLogfile(logger, filename: str | None = None):

    filename = testsettings.testResultsDir + "/" + filename

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    if isinstance(filename, str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.isfile(filename):
            os.remove(filename)
        with open(filename, 'x') as _:
            pass
        handler = logging.FileHandler(filename)
        logger.toFile = True
        logger.filename = filename
    else:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.toFile = False
        logger.filename = ""

    formatter = CustomFormatter(width=80)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def isLoggingEnabled(logger):
    if logger.isEnabledFor(logging.DEBUG):
        return True

    return False