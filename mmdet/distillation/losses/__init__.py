from .mgd import FeatureLoss
from .fgd import FGDLoss

from .mgd_ours import FeatureLossOurs
from .mgd_ours_bmse import FeatureLossOursBMSE

__all__ = [
    'FeatureLoss', 'FGDLoss', 'FeatureLossOurs', 'FeatureLossOursBMSE'
]
