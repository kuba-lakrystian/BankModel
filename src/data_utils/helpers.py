import dill

from src.data_utils.constants import *


class Serialization:
    @staticmethod
    def save_state(element, file_name, file_path):
        with open(
            SLASH_STR.join([file_path, DOT_STR.join([file_name, PICKLE])]), "wb"
        ) as dill_file:
            dill.dump(element, dill_file)

    @staticmethod
    def load_state(file_name, file_path):
        with open(
            SLASH_STR.join([file_path, DOT_STR.join([file_name, PICKLE])]), "rb"
        ) as dill_file:
            element = dill.load(dill_file)
            return element
