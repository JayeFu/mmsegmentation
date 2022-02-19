from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GateSegDataset(CustomDataset):
    """Gate Segmentation dataset.

    In segmentation map annotation for Gate Segmentation, 0 stands for gate, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('gate', 'background')

    PALETTE = [[0, 255, 0], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(GateSegDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
