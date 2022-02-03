"""TensorFlow preimport.  This code is to be imported before the TensorFlow
library is imported.
"""

# standard imports
import os

# toolbox imports
from dltb.config import config

# logging
from . import LOG

LOG.info("Preparing TensorFlow import")

# TF_CPP_MIN_LOG_LEVEL: Control the amount of TensorFlow log
# message displayed on the console.
#  0 = INFO
#  1 = WARNING
#  2 = ERROR
#  3 = FATAL
#  4 = NUM_SEVERITIES
# Defaults to 0, so all logs are shown.
LOG.info("Reduce amount of TensorFlow messages "
         "(set TF_CPP_MIN_LOG_LEVEL=1, i.e. only WARNING or more severe)")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# TF_CPP_MIN_VLOG_LEVEL brings in extra debugging information
# and actually works the other way round: its default value is
# 0 and as it increases, more debugging messages are logged
# in.
# Remark: VLOG messages are actually always logged at the INFO
# log level. It means that in any case, you need a
# TF_CPP_MIN_LOG_LEVEL of 0 to see any VLOG message.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

if config.use_cpu:  # use CPU even if GPU is available
    LOG.info("Disabling GPUs for TensorFlow (set CUDA_VISIBLE_DEVICES='')")
    # Using CUDA_VISIBLE_DEVICES environment variable indicates which
    # CUDA devices should be accessible to TensorFlow
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # first two GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # do not use GPU

    # CUDA_DEVICE_ORDER:
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # TF_FORCE_GPU_ALLOW_GROWTH:
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
