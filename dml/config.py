from util import Hparam, get


class Config:
    hps:Hparam = None

    @classmethod
    def init(cls):
        cls.hps = Hparam()