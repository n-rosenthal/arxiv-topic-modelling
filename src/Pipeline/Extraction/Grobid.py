"""src.Pipeline.Extraction.Grobid
Cliente mínimo para o serviço HTTP do GROBID.

O GROBID roda como um servidor à parte (normalmente em Docker, porta
8070). Este módulo envia um PDF para `/api/processFulltextDocument` e
devolve o TEI XML com metadados, seções e referências já estruturados.

Dois tipos de falha são distinguidos de propósito, porque pedem reações
opostas de quem chama:

    GrobidUnavailableError:
        o servidor não responde (caiu, não subiu, está saturado). Nada
        a ver com o documento; o lote inteiro deve parar em vez de
        marcar milhares de papers como "falhos".
    GrobidExtractionError:
        o servidor respondeu, mas não conseguiu processar *este*
        documento. Registra-se a falha e segue-se para o próximo.
"""

from __future__ import annotations

import logging
import os
import time

from pathlib import Path
from typing import Optional

import requests


logger = logging.getLogger(__name__)

DEFAULT_GROBID_URL = "http://localhost:8070"


class GrobidError(Exception):
    """Base para erros do cliente GROBID."""


class GrobidUnavailableError(GrobidError):
    """O servidor GROBID não está acessível ou está saturado."""


class GrobidExtractionError(GrobidError):
    """O GROBID não conseguiu extrair texto de um documento específico."""


class GrobidClient:
    """
    Args:
        base_url:
            Endereço do servidor. Padrão: variável de ambiente
            `GROBID_URL`, ou http://localhost:8070.
        timeout:
            Segundos de espera pela resposta de um documento. PDFs
            grandes podem levar dezenas de segundos.
        max_retries:
            Tentativas quando o servidor responde 503 (todas as
            threads ocupadas). Esgotadas, vira GrobidUnavailableError.
        retry_wait:
            Espera base, em segundos, entre tentativas após um 503
            (multiplicada pelo número da tentativa). Um cabeçalho
            `Retry-After` do servidor tem precedência.
        consolidate_header / consolidate_citations:
            Pedem ao GROBID para conferir metadados e referências em
            serviços externos (CrossRef etc.). Ficam desligados: são
            lentos, dependem de rede e os metadados do arXiv já estão
            no banco.
        segment_sentences:
            Pede ao GROBID para marcar sentenças (`<s>`) no TEI.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 180.0,
        max_retries: int = 5,
        retry_wait: float = 5.0,
        consolidate_header: bool = False,
        consolidate_citations: bool = False,
        segment_sentences: bool = False,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("GROBID_URL") or DEFAULT_GROBID_URL
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.consolidate_header = consolidate_header
        self.consolidate_citations = consolidate_citations
        self.segment_sentences = segment_sentences
        self.session = session or requests.Session()

    def is_alive(self) -> bool:
        """True se o servidor responde ao health check do GROBID."""
        try:
            response = self.session.get(f"{self.base_url}/api/isalive", timeout=5)
        except requests.RequestException:
            return False

        return response.status_code == 200 and response.text.strip().lower() == "true"

    def process_fulltext(self, pdf_path: Path | str) -> str:
        """
        Envia o PDF ao GROBID e retorna o TEI XML do texto completo.

        Raises:
            GrobidUnavailableError: servidor inacessível, ou ocupado
                (503) após `max_retries` tentativas.
            GrobidExtractionError: o servidor não conseguiu processar
                este documento (500, 204, 4xx, timeout).
        """
        path = Path(pdf_path)
        url = f"{self.base_url}/api/processFulltextDocument"

        data = {
            "consolidateHeader": "1" if self.consolidate_header else "0",
            "consolidateCitations": "1" if self.consolidate_citations else "0",
        }
        if self.segment_sentences:
            data["segmentSentences"] = "1"

        for attempt in range(1, self.max_retries + 1):
            try:
                # O arquivo é reaberto a cada tentativa: um POST anterior
                # já consumiu o ponteiro de leitura.
                with path.open("rb") as handle:
                    response = self.session.post(
                        url,
                        files={"input": (path.name, handle, "application/pdf")},
                        data=data,
                        headers={"Accept": "application/xml"},
                        timeout=self.timeout,
                    )
            except requests.ConnectionError as exc:
                # Inclui ConnectTimeout: o servidor não foi alcançado.
                raise GrobidUnavailableError(
                    f"cannot reach GROBID at {self.base_url}: {exc}"
                ) from exc
            except requests.Timeout as exc:
                raise GrobidExtractionError(
                    f"GROBID did not answer within {self.timeout}s for {path.name}"
                ) from exc
            except requests.RequestException as exc:
                raise GrobidExtractionError(
                    f"request to GROBID failed for {path.name}: {exc}"
                ) from exc

            status = response.status_code

            if status == 200:
                return response.content.decode("utf-8", errors="replace")

            if status == 503:
                wait = self._wait_before_retry(response, attempt)
                logger.info(
                    "GROBID busy (503) for %s; retry %d/%d in %.1fs",
                    path.name, attempt, self.max_retries, wait,
                )
                time.sleep(wait)
                continue

            if status == 204:
                raise GrobidExtractionError(
                    f"GROBID returned no content (204) for {path.name}"
                )

            raise GrobidExtractionError(
                f"GROBID returned HTTP {status} for {path.name}: "
                f"{response.text[:200].strip()}"
            )

        raise GrobidUnavailableError(
            f"GROBID still busy (503) after {self.max_retries} attempts"
        )

    def _wait_before_retry(self, response: requests.Response, attempt: int) -> float:
        header = response.headers.get("Retry-After")

        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass

        return self.retry_wait * attempt