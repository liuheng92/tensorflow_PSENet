import logging
from easydict import EasyDict as edict

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

__C = edict()
cfg = __C

#log level
__C.error = logging.ERROR
__C.warning = logging.WARNING
__C.info = logging.INFO
__C.debug = logging.DEBUG
