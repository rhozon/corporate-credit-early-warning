"""The seven model families compared in the paper, behind one interface."""
from .classic import Altman, HazardLogit
from .deep import DeepHit, DeepSurv
from .survival import RSF, Cox, XgbCox, XgbPD12

MODELS = {m.name: m for m in (Altman, HazardLogit, Cox, XgbPD12, XgbCox, RSF, DeepSurv, DeepHit)}
