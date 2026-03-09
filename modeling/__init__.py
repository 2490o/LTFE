from .rpn import SBRPN
from .backbone import ClipRN101
from .meta_arch_LTFE import ClipRCNNWithClipBackbone
from .roi_head_LTFE import ClipRes5ROIHeads
from .config import add_stn_config
from .custom_pascal_evaluation import CustomPascalVOCDetectionEvaluator