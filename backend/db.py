import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db"))

from db_utils import get_engine as get_app_engine  # noqa: F401
