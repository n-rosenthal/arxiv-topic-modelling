"""
src/Models/Processing/Stages.py

Defines the `ProcessingStage` enum for stages in the paper procesisng pipeline.
"""
from __future__ import annotations

__author__  = "n-rosenthal"
__version__ = "1.0.1"
__date__    = "2026-10-18"

from enum import Enum

class ProcessingStage(str, Enum):
    """Stages in the paper-processing pipeline.
    The stages will be updated as the pipeline's implementation proceeds.

    1. METADATA_EXTRACTION:
       arXiv[metadata] -> local[metadata]
       The attributes of an Paper were extracted through arXiv API.
    
    2. DOWNLOAD_PENDING
       (we can use the URL in the metadata for downloading the Paper's
       PDF, but this has not happened yet).

    3.  DOWNLOADING
        (we are downloading a Paper's PDF file).

    4.  PDF_AVAILABLE
        local[metadata] -> (id, local[PDF])
        The PDF for a Paper is available, but has not yet been validated

    5.  PDF_VALIDATION
        local[PDF] -> local[ValidatedPDF]
        Applies validation methods over the Paper's PDF.

    6.  PDF_TO_TEXT
        local[ValidatedPDF] -> local[Text]

    7.  TEXT_AVAILABLE
        (The Paper's PDF was processed to generate its plain text,
        but this has not yet been processed).


    (-2.) INVALID_DOWNLOAD
        local[metadata] -> Error
        Couldn't download Paper's PDF.
    
    (-5.) INVALID_PDF
        local[PDF] -> Error
        Current Paper's PDF is invalid and cannot be further processed nor used.

    (-6.) INVALID_TEXT
        local[PDF] -> Error
        Couldn't extract valid text from the Paper's PDF.

    
    
    
    """
    METADATA_EXTRACTION = "metadata"

    DOWNLOAD_PENDING    = "download_pending"
    DOWNLOADING         = "downloading"

    PDF_AVAILABLE       = "pdf_available"
    PDF_VALIDATION      = "pdf_validation"
    PDF_TO_TEXT         = "pdf_to_text"

    TEXT_AVAILABLE      = "text_available"


    # Error stages
    INVALID_DOWNLOAD    = "invalid_download"
    INVALID_PDF         = "invalid_pdf"
    INVALID_TEXT        = "invalid_text"
    
#    TEXT_EXTRACTION = "text_extraction"
#    PREPROCESSING = "preprocessing"
#    EMBEDDING = "embedding"
#    GRAPH_CONSTRUCTION = "graph_construction"
#    TOPIC_MODELING = "topic_modeling"
