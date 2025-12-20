"""
call clean_empty_labels() to remove empty label files and their corresponding images.
Only use if the subject is always in view.

"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clean_empty_labels import clean_empty_labels

clean_empty_labels('data/sam_output/labels', 'data/sam_output/images')

print("Script says: Empty label cleaning complete.")