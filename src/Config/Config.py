"""
src/Config/Config.py

Configuration for the project
"""

__author__  = "n-rosenthal"
__version__ = "1.0.1"
__date__    = "2026-10-18"

from pathlib import Path


config: dict[str, str
                | list[str]
                | Path] = {
    # paths
    "PROJECT_ROOT"    : Path(__file__).resolve().parents[1],
    "DATA_DIR"        : Path(__file__).resolve().parents[1] / "data",
    "DATA_CORPUS_DIR" : Path(__file__).resolve().parents[1] / "data" / "corpus",
                    
    # arXiv, PDF downloading
    "DEFAULT_DATASET_ID": "CShorten/ML-ArXiv-Papers",
    "DEFAULT_SPLIT":      "train",
    
    # arXiv client
    "ARXIV_CLIENT_PAGE_SIZE"     : "100",
    "ARXIV_CLIENT_DELAY_SECONDS" : "6.0",
    "ARXIV_CLIENT_MAX_RETRIES"   : "3",
    "ARXIV_CLIENT_USER_AGENT"    : "arxiv-topic-modelling // academic research project in NLP // (Python urllib)",
    
    
    # arXiv categories and time intervals
    "DEFAULT_CATEGORIES": [
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.LG",
        "cs.ML",
        "cs.NE",
        "cs.NI",
        "stat.ML",
    ],
    "DEFAULT_CATEGORY":   "cs.LG",
    
    
    "DEFAULT_START_DATE": "20250101",
    "DEFAULT_END_DATE":   "20251231",
}

def category_name(category: str) -> str:
    match category:
        case "cs.AI":
            return "Artificial Intelligence"
        case "cs.CL":
            return "Computation and Language"
        case "cs.CV":
            return "Computer Vision and Pattern Recognition"
        case "cs.LG":
            return "Logic and Games"
        case "cs.ML":
            return "Machine Learning"
        case "cs.NE":
            return "Neural and Evolutionary Computing"
        case "cs.NI":
            return "Neuroinformatics"
        case "stat.ML":
            return "Statistics and Machine Learning"
        case _:
            raise ValueError(f"Unknown category: {category}")
