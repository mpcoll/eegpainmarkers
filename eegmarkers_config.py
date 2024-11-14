# -*- coding:utf-8 -*-
# @Script: painexplo_config.py
# @Description: Global parameters and helper functions
# for the pain exploration analyses

from os.path import join as opj
import os
import numpy as np


# Global parameters
class global_parameters:
    def __init__(self):
        # Path
        self.bidspath = "/Volumes/eeg_ml/2024_eegpainmarkers"  # Path to BIDS directory

        # Ressources
        self.ncpus = 10  # Number of CPUs to use
        self.mem_gb = 64  # Memory in GB
