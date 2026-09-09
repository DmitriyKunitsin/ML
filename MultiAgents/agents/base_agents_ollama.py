import time
import threading
import sys
from utils.ollama_client import ask_ollama


class BaseAgentsOllama:
    """Базовый класс агента, который умеет базовые вещи для агента"""

    def __init__(self, model, task_type, config):
        self.model = model
        self.task_type = task_type
        self.config = config

    def ask(self, promt, system_promt=""):
        """Отправляет запрос к Ollama через общий клиент"""
        return ask_olla
