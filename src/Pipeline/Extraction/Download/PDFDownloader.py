"""
Download individual arXiv papers as PDF files.

Downloads papers from the official arXiv PDF endpoint, validates the
downloaded files, and records artifact metadata in the Paper model.
"""

"""Download and validate individual papers from arXiv."""
from src.Models import (
    ProcessingStage,
    DownloadInfo,
    Paper,
)

import hashlib
import logging
import os
import re
import tempfile
import time

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import quote

logger = logging.getLogger(__name__)


def directory_size(dir: Path) -> int:
    if not dir.exists():
        return 0
    return sum(f.stat().st_size for f in dir.iterdir() if f.is_file())


def file_size(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def str_download(arxiv_id: str) -> str:
    pdf_dir = Path("data/pdfs")
    sample_file = pdf_dir / "2301.00001.pdf"
    return (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
        f"[dir=`{directory_size(pdf_dir)}`] "
        f"[file=`{file_size(sample_file)}`] "
        f"`{arxiv_id}` . . ."
    )


class PDFDownloadError(RuntimeError):
    """Raised when a paper cannot be downloaded or validated."""


class PDFDownloader:
    """Download individual PDF files from the arXiv export service.

    Downloads are sequential and rate-limited. Files are written to a
    temporary file and atomically moved into place only after validation.

    Parameters
    ----------
    output_dir:
        Directory where downloaded PDFs are stored.
    overwrite:
        If True, download again even when a valid PDF already exists.
    timeout:
        Timeout in seconds for each HTTP request.
    max_retries:
        Number of retries after the initial request.
    request_interval:
        Minimum delay between the start of consecutive HTTP requests.
        A conservative default of 3 seconds is used.
    chunk_size:
        Number of bytes read per iteration.
    """

    BASE_URL = "https://export.arxiv.org/pdf/"

    USER_AGENT = (
        "arxiv-topic-modelling/0.1 "
        "(academic research; Python urllib)"
    )

    # Modern IDs, e.g. 2301.00001 or 0902.3253.
    # Legacy IDs, e.g. hep-th/9901001 or math.GT/0309136.
    ARXIV_ID_PATTERN = re.compile(
        r"^(?:"
        r"\d{4}\.\d{4,5}"
        r"|"
        r"[\w.-]+/\d{7}"
        r")"
        r"(?:v\d+)?$"
    )

    PDF_SIGNATURE = b"%PDF-"

    def __init__(
        self,
        output_dir: str | Path = "data/pdfs",
        overwrite: bool = False,
        timeout: float = 60.0,
        max_retries: int = 4,
        request_interval: float = 5.0,
        chunk_size: int = 1024 * 128,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")

        if request_interval < 0:
            raise ValueError("request_interval must be non-negative.")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_interval = request_interval
        self.chunk_size = chunk_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Used to throttle requests made by this downloader instance.
        self._last_request_at: float | None = None

    @classmethod
    def _validate_arxiv_id(cls, arxiv_id: str) -> str:
        """Validate and normalize an arXiv identifier."""
        arxiv_id = arxiv_id.strip()

        if not cls.ARXIV_ID_PATTERN.fullmatch(arxiv_id):
            raise ValueError(
                f"Invalid arXiv ID: {arxiv_id!r}"
            )

        return arxiv_id

    @staticmethod
    def _safe_filename(arxiv_id: str) -> str:
        """Convert an arXiv ID into a safe local filename."""
        return arxiv_id.replace("/", "_") + ".pdf"

    def _output_path(self, arxiv_id: str) -> Path:
        return self.output_dir / self._safe_filename(arxiv_id)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute a file's SHA-256 hash without loading it all into RAM."""
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @classmethod
    def _is_valid_pdf(cls, path: Path) -> bool:
        """Check whether a non-empty file has a PDF signature."""
        try:
            if not path.is_file() or path.stat().st_size == 0:
                return False

            with path.open("rb") as file:
                header = file.read(1024)

            # PDF files normally start with %PDF-. Allow a short prefix
            # because some otherwise valid files have leading bytes.
            return cls.PDF_SIGNATURE in header

        except OSError:
            return False

    def _wait_for_rate_limit(self) -> None:
        """Ensure a minimum interval between request starts."""
        if self._last_request_at is None:
            return

        elapsed = time.monotonic() - self._last_request_at
        remaining = self.request_interval - elapsed

        if remaining > 0:
            time.sleep(remaining)

    def _build_url(self, arxiv_id: str) -> str:
        # Preserve "/" for legacy identifiers such as hep-th/9901001.
        encoded_id = quote(arxiv_id, safe="/.")
        return f"{self.BASE_URL}{encoded_id}"

    @staticmethod
    def _retry_after_seconds(error: HTTPError) -> float | None:
        """Read Retry-After if the server provides it."""
        value = error.headers.get("Retry-After")

        if not value:
            return None

        try:
            return max(0.0, float(value))
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(value)

            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            return max(0.0, (retry_at - now).total_seconds())

        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _is_retryable_http_status(status: int) -> bool:
        """Return whether an HTTP status may succeed on a later attempt."""
        return status in {
            408,  # Request Timeout
            425,  # Too Early
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        }

    def _download_to_file(
        self,
        url: str,
        destination: Path,
    ) -> tuple[int, str | None, str]:
        """Download one response into a temporary file.

        Returns
        -------
        tuple
            File size, content type, and SHA-256 hash.
        """
        request = Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "application/pdf, application/octet-stream;q=0.9",
            },
        )

        self._wait_for_rate_limit()
        self._last_request_at = time.monotonic()

        digest = hashlib.sha256()
        total_bytes = 0
        header = bytearray()

        with urlopen(request, timeout=self.timeout) as response:
            content_type = response.headers.get("Content-Type")

            with destination.open("wb") as output:
                while True:
                    chunk = response.read(self.chunk_size)

                    if not chunk:
                        break

                    # Retain only enough initial bytes for signature checking.
                    if len(header) < 1024:
                        remaining = 1024 - len(header)
                        header.extend(chunk[:remaining])

                    output.write(chunk)
                    digest.update(chunk)
                    total_bytes += len(chunk)

        if total_bytes == 0:
            raise PDFDownloadError(
                f"The server returned an empty file for {url}."
            )

        if self.PDF_SIGNATURE not in header:
            raise PDFDownloadError(
                f"The response from {url} does not appear to be a PDF."
            )
            
        return total_bytes, content_type, digest.hexdigest()

    def _download_with_retries(
        self,
        url: str,
        destination: Path,
    ) -> tuple[int, str | None, str]:
        """Download a PDF, retrying transient network and HTTP failures."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._download_to_file(url, destination)

            except HTTPError as error:
                last_error = error

                if not self._is_retryable_http_status(error.code):
                    raise PDFDownloadError(
                        f"Non-retryable HTTP {error.code} while "
                        f"downloading {url}: {error.reason}"
                    ) from error

                retry_after = self._retry_after_seconds(error)

                if attempt == self.max_retries:
                    break

                # Exponential backoff with a modest upper bound.
                delay = min(60.0, 2.0 ** attempt)

                if retry_after is not None:
                    delay = max(delay, retry_after)

                logger.warning(
                    "HTTP %s downloading %s; retrying in %.1f seconds "
                    "(attempt %d/%d).",
                    error.code,
                    url,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )

                time.sleep(delay)

            except (URLError, TimeoutError, OSError) as error:
                last_error = error

                if attempt == self.max_retries:
                    break

                delay = min(60.0, 2.0 ** attempt)

                logger.warning(
                    "Network error downloading %s: %s. "
                    "Retrying in %.1f seconds (attempt %d/%d).",
                    url,
                    error,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )

                time.sleep(delay)

            except PDFDownloadError as e:
                # A non-PDF response should not be retried blindly.
                raise e

        raise PDFDownloadError(
            f"Failed to download {url} after "
            f"{self.max_retries + 1} attempts."
        ) from last_error

    def _record_download(
        self,
        paper: Paper,
        path: Path,
        url: str,
        size: int,
        sha256: str,
        content_type: str | None,
    ) -> None:
        """Record the download artifact and complete its processing stage."""
        paper.processing.download = DownloadInfo(
            local_path=str(path),
            file_size_bytes=size,
            sha256=sha256,
            downloaded_at=datetime.now(timezone.utc),
            content_type=content_type,
            source_url=url,
            verified=True,
        )

        paper.update_timestamp()

    def download(
        self,
        paper: Paper,
        overwrite: bool | None = None,
    ) -> Path:
        """Download a paper's PDF and return its local path.

        A valid existing file is reused unless overwrite is True.
        Invalid existing files are replaced by a new validated download.
        """
        arxiv_id = self._validate_arxiv_id(paper.arxiv_id)
        output_path = self._output_path(arxiv_id)
        url = self._build_url(arxiv_id)

        force_download = (
            self.overwrite if overwrite is None else overwrite
        )

        try:
            if output_path.exists() and not force_download:
                if self._is_valid_pdf(output_path):
                    size = output_path.stat().st_size
                    sha256 = self._sha256_file(output_path)

                    logger.info(
                        "Reusing existing PDF for %s: %s",
                        arxiv_id,
                        output_path,
                    )

                    self._record_download(
                        paper=paper,
                        path=output_path,
                        url=url,
                        size=size,
                        sha256=sha256,
                        content_type="application/pdf",
                    )

                    return output_path

                logger.warning(
                    "Existing PDF for %s is invalid; downloading again.",
                    arxiv_id,
                )

            logger.info(
                "Downloading arXiv paper %s from %s",
                arxiv_id,
                url,
            )

            # Use a temporary file in the destination directory so that
            # os.replace() is atomic on the same filesystem.
            temporary_path: Path | None = None

            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{output_path.stem}.",
                    suffix=".part",
                    dir=self.output_dir,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)

                size, content_type, sha256 = self._download_with_retries(
                    url=url,
                    destination=temporary_path,
                )

                if not self._is_valid_pdf(temporary_path):
                    raise PDFDownloadError(
                        f"Downloaded file failed PDF validation: {arxiv_id}"
                    )

                os.replace(temporary_path, output_path)
                temporary_path = None

                self._record_download(
                    paper=paper,
                    path=output_path,
                    url=url,
                    size=size,
                    sha256=sha256,
                    content_type=content_type,
                )

                logger.info(
                    "Downloaded %s (%d bytes) to %s",
                    arxiv_id,
                    size,
                    output_path,
                )
                
                print(str_download(arxiv_id))

                return output_path

            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        except Exception as error:
            if isinstance(error, (ValueError, PDFDownloadError)):
                raise

            raise PDFDownloadError(
                f"Could not download arXiv paper {arxiv_id}: {error}"
            ) from error
