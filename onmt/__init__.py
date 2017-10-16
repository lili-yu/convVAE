import onmt.IO
import onmt.Modelsvae
import onmt.Loss
from onmt.VaeTrainer import VaeTrainer, Statistics
from onmt.Translator import Translator


from onmt.Optim import Optim
from onmt.Beam import Beam

from onmt.SRU import check_sru_requirement
can_use_sru = check_sru_requirement()
if can_use_sru:
    from onmt.SRU import SRU

from onmt.UtilClass import LayerNorm, Bottle, BottleLinear, \
    BottleLayerNorm, BottleSoftmax, Elementwise
from onmt.StackedRNN import StackedLSTM, StackedGRU
from onmt.Embeddings import Embeddings    

# For flake8 compatibility
__all__ = [onmt.Loss, onmt.IO, VaeTrainer, Translator,
           Optim, Beam, Statistics, Embeddings, Elementwise,StackedLSTM, StackedGRU]


if can_use_sru:
    __all__.extend([SRU, check_sru_requirement])