import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pages.Data_visualization_function import *
from Data_for_test import raw_data, clean_data

import pytest 
from streamlit.testing.v1 import AppTest
