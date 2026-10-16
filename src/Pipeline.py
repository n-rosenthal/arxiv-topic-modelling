"""
Paper acquisition and text-extraction pipeline.
"""
from pathlib import Path

from src.Extractors.PDFDownloader import PDFDownloader
from src.Extractors.PDFTextExtractor import PDFTextExtractor
from src.Models.Paper import Paper


class PaperPipeline:
    """Download and extract the text of an arXiv paper."""

    def __init__(
        self,
        pdf_dir: str | Path = "data/pdfs",
        text_dir: str | Path = "data/texts",
        overwrite_pdfs: bool = False,
        overwrite_text: bool = False,
    ) -> None:
        self.pdf_downloader = PDFDownloader(
            output_dir=pdf_dir,
            overwrite=overwrite_pdfs,
        )

        self.text_extractor = PDFTextExtractor(
            output_dir=text_dir,
            overwrite=overwrite_text,
        )

    def process(self, paper: Paper) -> Paper:
        """Download a paper, extract its text, and return its updated model."""
        pdf_path = self.pdf_downloader.download(paper)

        self.text_extractor.extract(
            paper,
            pdf_path=pdf_path,
        )

        return paper