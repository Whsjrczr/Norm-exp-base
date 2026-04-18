import importlib

from .modules import *
from .my_modules import *

from .progress_bar import ProgressBar
from .trainer import *
from . import scheduler, optimizer, vis_taiyi
from . import logger, visdom, tracking, taiyi, checkpoint
from . import dataset, trainer
from .activation import *
from .multichannel import *
from .normalization import *

activation = importlib.import_module(".activation", __name__)
checkpoint = importlib.import_module(".checkpoint", __name__)
dataset = importlib.import_module(".dataset", __name__)
logger = importlib.import_module(".logger", __name__)
measurement = importlib.import_module(".measurement", __name__)
model = importlib.import_module(".model", __name__)
multichannel = importlib.import_module(".multichannel", __name__)
normalization = importlib.import_module(".normalization", __name__)
ntk = importlib.import_module(".ntk", __name__)
optimizer = importlib.import_module(".optimizer", __name__)
scheduler = importlib.import_module(".scheduler", __name__)
taiyi = importlib.import_module(".taiyi", __name__)
tracking = importlib.import_module(".tracking", __name__)
trainer = importlib.import_module(".trainer", __name__)
vis_taiyi = importlib.import_module(".vis_taiyi", __name__)
visdom = importlib.import_module(".visdom", __name__)
