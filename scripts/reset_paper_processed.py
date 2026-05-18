import os
from pathlib import Path
import sys

 # This is a workaround for now to be able to import from src/ until we package this properly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.serialization import load_papers_from_json, save_paper
from src.data.parser import parse_pdf_for_pages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/papers" 

if __name__ == "__main__":
    for paper in load_papers_from_json():
        paper.processed = False
        save_paper(paper)

