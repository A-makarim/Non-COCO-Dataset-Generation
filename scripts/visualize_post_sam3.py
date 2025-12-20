"""
visualise images with annotated labels
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.annotate_labels_on_images import annotate_labels_on_images


def main():
    annotate_labels_on_images()
    print("Script says: Annotated images have been saved to the output directory.")


if __name__ == "__main__":
    main()