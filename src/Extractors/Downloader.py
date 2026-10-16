"""
src/Extractors/Downloader.py

Provides the ArxivDownloader class for retrieving selected records
from a Hugging Face dataset of arXiv papers.

Records are streamed and written to JSON Lines (JSONL), avoiding the
need to load the entire dataset into memory.
"""

__author__ = "n-rosenthal"
__version__ = "1.0.0"
__date__ = "2026-10-15"

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

from datasets import IterableDataset, load_dataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ArxivDownloader:
    """
    Download selected records from a Hugging Face arXiv dataset.

    Parameters
    ----------
    dataset_id : str
        Hugging Face dataset repository identifier.
    split : str
        Dataset split to retrieve.
    cache_dir : str or Path, optional
        Directory used for Hugging Face's local cache.
    """

    DEFAULT_DATASET_ID = "CShorten/ML-ArXiv-Papers"
    DEFAULT_SPLIT = "train"

    def __init__(
        self,
        dataset_id: str = DEFAULT_DATASET_ID,
        split: str = DEFAULT_SPLIT,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        if not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty.")

        if not split.strip():
            raise ValueError("split cannot be empty.")

        self.dataset_id = dataset_id
        self.split = split
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else None
        )

    def load(self) -> IterableDataset:
        """Load metadata from the configured Hugging Face dataset."""
        from datasets import load_dataset

        logger.info(
            "Loading dataset %r, split %r.",
            self.dataset_id,
            self.split,
        )

        cache_dir = (
            str(self.cache_dir)
            if self.cache_dir is not None
            else None
        )

        if self.dataset_id == "nick007x/arxiv-papers":
            parquet_url = (
                "https://huggingface.co/datasets/"
                f"{self.dataset_id}/resolve/main/{self.split}.parquet"
            )

            logger.info(
                "Loading metadata from Parquet: %s",
                parquet_url,
            )

            return load_dataset(
                "parquet",
                data_files={self.split: parquet_url},
                split=self.split,
                streaming=True,
                cache_dir=cache_dir,
            )

        return load_dataset(
            self.dataset_id,
            split=self.split,
            streaming=True,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _select_range(
        dataset: IterableDataset,
        start: int,
        limit: Optional[int],
    ) -> Iterator[dict[str, Any]]:
        """
        Select a range of records from a streaming dataset.

        Parameters
        ----------
        dataset : IterableDataset
            Source dataset.
        start : int
            Zero-based index of the first requested record.
        limit : int or None
            Maximum number of records to retrieve.
            None means continue until the split is exhausted.
        """
        selected = dataset.skip(start)

        if limit is not None:
            selected = selected.take(limit)

        yield from selected

    def download(
        self,
        output_path: str | Path,
        start: int = 0,
        limit: Optional[int] = 100,
        overwrite: bool = False,
    ) -> int:
        """
        Download a selected range of papers into a JSONL file.

        Parameters
        ----------
        output_path : str or Path
            Destination file.
        start : int
            Zero-based index of the first record.
        limit : int or None
            Maximum number of records. None downloads the remainder
            of the split.
        overwrite : bool
            Whether to replace an existing output file.

        Returns
        -------
        int
            Number of records written.

        Raises
        ------
        ValueError
            If start or limit is invalid.
        FileExistsError
            If the output file exists and overwrite is False.
        """
        if start < 0:
            raise ValueError("start must be non-negative.")

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None.")

        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Pass overwrite=True to replace it."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset = self.load()
        records = self._select_range(
            dataset=dataset,
            start=start,
            limit=limit,
        )

        count = 0

        logger.info(
            "Writing records to '%s'.",
            output_path,
        )

        try:
            with output_path.open(
                mode="w",
                encoding="utf-8",
            ) as output_file:
                with tqdm(
                    total=limit,
                    desc="Downloading metadata",
                    unit="record",
                ) as progress:
                    for record in records:
                        output_file.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                default=str,
                            )
                            + "\n"
                        )

                        count += 1
                        progress.update(1)

        except Exception:
            # Do not leave a partial output looking like a
            # successfully completed download.
            output_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Downloaded %d records to '%s'.",
            count,
            output_path,
        )

        return count