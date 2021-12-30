# -*- coding: utf-8 -*-
"""
=====================================
# Time        : 2020/9/9  11:47
# File        : _logger.py
# Author      : Wang Xiaoyu
=====================================
Intro 
"""

import logging

# create logger
logger = logging.getLogger('causalRL_logger')
logger.setLevel(logging.DEBUG)

# create handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# ch.setLevel(logging.DEBUG)

# create formatter
# formatter = logging.Formatter('%(filename)s - %(levelname)-8s - %(message)s')
# formatter = logging.Formatter('%(filename)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

if __name__ == '__main__':

    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.critical('critical message')
    # logger.info('1', '3')


