from garage.torch.algos.mtsac import MTSAC
from nupic.embodied.algos.sparse_weights_sac import SparseWeightsSAC


class SparseWeightsMTSAC(MTSAC, SparseWeightsSAC):
    pass
