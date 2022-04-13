"""
This module implements a logger to be used across projects.
"""
import logging


class Logger:
    """
    The default logger for python projects
    """
    def __init__(self):
        self.logger = logging
