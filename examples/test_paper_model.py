"""Demonstrate the Paper data model."""

import json

from src.Models import Paper, ProcessingStage, ProcessingStatus


def main() -> None:
    # Representative source record.
    record = {
        "arxiv_id": "2301.00001",
        "title": "Example Paper on Graph Neural Networks",
        "authors": [
            "Alice Smith",
            "Bob Jones",
        ],
        "submission_date": "18 Feb 2009",
        "comments": "12 pages, 4 figures",
        "primary_subject": "Computer Science - Machine Learning",
        "subjects": "cs.LG cs.AI",
        "doi": None,
        "abstract": (
            "This paper investigates graph representations "
            "for scientific document collections."
        ),
        "file_path": "example/path/to/paper.pdf",
    }

    paper = Paper.from_source_record(
        record,
        dataset_id="nick007x/arxiv-papers",
        dataset_split="train",
        source_record_index=0,
    )

    print(f"Paper: {paper.title}")
    print(f"arXiv ID: {paper.arxiv_id}")
    print(f"PDF downloaded: {paper.pdf_downloaded}")
    print(f"Text processed: {paper.text_processed}")

    # Simulate a successful PDF download.
    download_state = paper.processing.state(
        ProcessingStage.PDF_DOWNLOAD
    )
    download_state.start()
    download_state.complete()

    paper.processing.download.local_path = (
        "data/raw/pdfs/2301.00001.pdf"
    )
    paper.processing.download.verified = True

    # Simulate successful text preprocessing.
    preprocessing_state = paper.processing.state(
        ProcessingStage.PREPROCESSING
    )
    preprocessing_state.start()
    preprocessing_state.complete()

    paper.processing.processed_text.local_path = (
        "data/processed/2301.00001.txt"
    )
    paper.processing.processed_text.character_count = 2500

    paper.update_timestamp()

    print()
    print(f"PDF downloaded: {paper.pdf_downloaded}")
    print(f"Text processed: {paper.text_processed}")

    print()
    print("Serialized paper:")
    print(json.dumps(paper.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()