import logging
import graypy

logger = logging.getLogger("test_logger")
logger.setLevel(logging.DEBUG)

handler = graypy.GELFTCPHandler('localhost', 8384)
logger.addHandler(handler)

logger.debug("hello graylog 2")
