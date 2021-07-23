import logging
from abc import ABC, abstractmethod


class RLAlgorithhm(ABC):

    @abstractmethod
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            log_formatter = logging.Formatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s')
            file_handler = logging.FileHandler('info.log')
            file_handler.setFormatter(log_formatter)
            self.logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    @abstractmethod
    def train(self, training_episodes):
        pass

    @abstractmethod
    def run(self, episodes, render):
        pass
