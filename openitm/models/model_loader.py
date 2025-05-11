# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from importlib import import_module


class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL_USE
        model_moudle_path = 'openitm.models.' + self.model_use + '.full_itm'
        self.model_moudle = import_module(model_moudle_path)

    def Net_Full(self, __arg1, __arg2):
        return self.model_moudle.Net_Full(__arg1, __arg2)
