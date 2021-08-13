import logging
from abc import ABC, abstractmethod

import numpy as np


class RLAlgorithhm(ABC):

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            log_formatter = logging.Formatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s')
            file_handler = logging.FileHandler('info.log')
            file_handler.setFormatter(log_formatter)
            self._logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            self._logger.addHandler(console_handler)
            self._logger.setLevel(logging.INFO)

    @abstractmethod
    def train(self, training_episodes: int) -> None:
        pass

    @abstractmethod
    def run(self, episodes: int, render: bool) -> None:
        pass
