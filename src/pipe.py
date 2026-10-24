"""
Paper acquisition and text-extraction pipeline.

Ciclo por paper (`PaperPipeline.process_paper`):

    arxiv.Search  ->  manifesto + banco             (CorpusBuilder)
    PDF           ->  download                      (PDFDownloader)
    TEI           ->  GROBID, gravado em `extracted_dir`   (ScientificPaperProcessor.extract)
    seções        ->  limpeza -> tokens             (ScientificPaperProcessor.preprocess)
    resultado     ->  JSON de seções em `sections_dir`, registrado no banco
    PDF local     ->  removido

O banco SQLite é a fila de trabalho: cada execução retoma de onde a
anterior parou, sem refazer download nem extração, e pode ser
interrompida (Ctrl+C) a qualquer momento.

O TEI é o artefato bruto durável. Mudou o limpador, as stopwords ou a
lematização? `reprocess()` (`--reprocess`) refaz limpeza e tokens a partir
dos TEI salvos, sem PDF e sem GROBID.
"""
from __future__ import annotations

import argparse

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from src.Pipeline.Extraction    import (
    CorpusBuilder,
    CorpusRecord,

    PDFDownloader,
)
from src.Pipeline.Extraction.Grobid import GrobidClient, GrobidUnavailableError
from src.Models.Paper           import Paper
from src.Models.Document        import ExtractedDocument

from src.Models.Processing import (
    ProcessingStatus, ProcessingStage, StageState
)
from src.Storage import Database

if TYPE_CHECKING:
    from src.Processing.ScientificPaperProcessor import ScientificPaperProcessor

def log(message: str) -> None:
    print(message)

class PipelineError(Exception):
    """Raised in processing pipeline errors."""
    pass


# Menos texto limpo que isto é tratado como extração falha: tipicamente
# um PDF escaneado (só imagens), do qual o GROBID devolve pouco ou nada.
MIN_TEXT_CHARS = 200


class PaperPipeline:
    """Download and extract the text of an arXiv paper."""

    def __init__(
        self,
        pdf_dir: str | Path = "data/pdfs",
        sections_dir: str | Path = "data/sections",
        overwrite_pdfs: bool = False,
        overwrite_text: bool = False,
        db_path: str | Path = Database.DEFAULT_DB_PATH,
        grobid: Optional[GrobidClient] = None,
        processor: Optional[ScientificPaperProcessor] = None,
        extracted_dir: str | Path = "data/extracted",
        spacy_model: Optional[str] = None,
        keep_pdfs: bool = False,
        max_attempts: int = 3,
        max_consecutive_failures: int = 10,
    ) -> None:
        self.pdf_dir = Path(pdf_dir)
        self.sections_dir = Path(sections_dir)
        self.overwrite_pdf = overwrite_pdfs
        self.overwrite_text = overwrite_text

        #   usado só para checar se o servidor está de pé (início do lote e
        #   após uma falha); a extração em si é do ScientificPaperProcessor
        self.grobid = grobid or GrobidClient()

        #   criado sob demanda (ver `processor`): `--download-only` não precisa dele
        self._processor = processor
        self._processor_kwargs = {
            "extracted_dir": extracted_dir,
            "sections_dir": sections_dir,
            "spacy_model": spacy_model,
        }

        #   True: mantém o PDF depois de extrair o texto
        self.keep_pdfs = keep_pdfs

        #   falhas (de download ou de extração) toleradas por paper
        #   antes de ele sair da fila de trabalho
        self.max_attempts = max_attempts

        #   falhas seguidas que interrompem o lote: um problema sistêmico
        #   (GROBID mal configurado, rede fora, bug) falharia a fila toda e
        #   gastaria as tentativas de milhares de papers de uma vez
        self.max_consecutive_failures = max_consecutive_failures

        #   banco de consulta de estado (quais papers já têm PDF/texto)
        self.db_path = Path(db_path)
        self.db = Database.get_connection(self.db_path)
        Database.init_db(self.db)

        #   searchs @ arxiv API for papers
        #   records them in a JSONL manifest
        self.extractor = CorpusBuilder(db_path=self.db_path)

        #   all the records and respective papers
        self.record_corpus: Optional[list[CorpusRecord]] = None
        self.papers: Optional[list[Paper]] = None

        #   downloads PDF for given papers
        self.downloader: PDFDownloader = PDFDownloader(
            output_dir=pdf_dir,
            overwrite=overwrite_pdfs,
            timeout=6,
            max_retries=3,
            request_interval=6,
        )

    @property
    def processor(self) -> ScientificPaperProcessor:
        """GROBID + limpeza + NLP, criado na primeira vez que é preciso."""
        if self._processor is None:
            from src.Processing.ScientificPaperProcessor import ScientificPaperProcessor

            self._processor = ScientificPaperProcessor(
                grobid_url=self.grobid.base_url,
                **self._processor_kwargs,
            )

        return self._processor

    def close(self) -> None:
        if self._processor is not None:
            self._processor.close()
        self.db.close()

    def __enter__(self) -> "PaperPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def corpus(self, limit: Optional[int]) -> list[CorpusRecord]:
        if self.record_corpus is None:
            self.record_corpus = self.extractor.build_corpus(limit if limit is not None else 100)
            return self.record_corpus
        return self.record_corpus

    def pending_papers(self, papers: list[Paper]) -> list[Paper]:
        """
        Filtra, dentre `papers`, apenas os que ainda precisam de
        download: sem texto extraído e sem PDF em disco.

        Papers com texto já extraído não são pendentes, mesmo sem PDF
        — ele é removido de propósito depois da extração.
        """
        Database.reconcile_pdfs(self.db)

        pending_ids = set(
            Database.get_pending_pdf_downloads(
                self.db,
                [p.arxiv_id for p in papers],
                max_attempts=self.max_attempts,
            )
        )

        return [p for p in papers if p.arxiv_id in pending_ids]

    def _get_paper(self, record: CorpusRecord, source_record_index: int) -> Optional[Paper]:
        return self.extractor.from_record(record, source_record_index=source_record_index)

    def get_papers(self, result_corpus: Optional[list[CorpusRecord]]) -> Optional[list[Paper]]:
        if result_corpus is None:
            return None

        papers: list[Paper] = []

        for i, r in enumerate(result_corpus):
            try:
                paper = self._get_paper(r, source_record_index=i)
                if paper is not None:
                    papers.append(paper)
            except Exception as e:
                log(f"Skipping record {i}: {e}")
                continue

        return papers

    def _paper_from_id(self, arxiv_id: str) -> Optional[Paper]:
        """
        Reconstrói o `Paper` de um arxiv_id a partir do banco.

        O `source_record_index` é irrelevante aqui: este Paper é
        transitório e só alimenta o downloader.
        """
        record = Database.get_record(self.db, arxiv_id)

        if record is None:
            return None

        return self.extractor.from_record(record, source_record_index=0)

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def download_pdf(self, paper: Paper) -> Path:
        """downloads the `pdf` for a Paper and returns the path to it"""
        try:
            path = self.downloader.download(paper, self.overwrite_pdf)
            Database.mark_pdf_downloaded(self.db, paper.arxiv_id, path)
            return path
        except Exception as e:
            log(
                f"Error downloading PDF for {paper.arxiv_id}: {e}",
            )
            Database.mark_pdf_failed(self.db, paper.arxiv_id, str(e))
            return Path("")

    def download_pending(self, limit: Optional[int] = None) -> dict[str, int]:
        """
        Baixa os PDFs dos papers que ainda não os têm (e cujo texto
        ainda não foi extraído), sem chamar o GROBID.

        Útil para formar um estoque de PDFs antes de extrair, ou para
        extrair depois com `process_pending`. Retomável: o que já está
        em disco não é baixado de novo.
        """
        Database.reconcile_pdfs(self.db)

        queue = Database.get_pending_pdf_downloads(
            self.db, max_attempts=self.max_attempts
        )
        if limit is not None:
            queue = queue[:limit]

        stats = {"downloaded": 0, "failed": 0}
        log(f"{len(queue)} PDF(s) to download")

        for i, arxiv_id in enumerate(queue, start=1):
            paper = self._paper_from_id(arxiv_id)
            if paper is None:
                continue

            log(f"[{i}/{len(queue)}] downloading `{arxiv_id}`")
            path = self.download_pdf(paper)
            stats["downloaded" if path.is_file() else "failed"] += 1

        log(f"done: {stats['downloaded']} downloaded, {stats['failed']} failed")
        return stats

    def _remove_pdf(self, arxiv_id: str, pdf_path: Path) -> bool:
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError as e:
            log(f"Could not remove {pdf_path}: {e}")
            return False

        Database.mark_pdf_removed(self.db, arxiv_id)
        return True

    # ------------------------------------------------------------------
    # Texto (GROBID + limpeza + NLP)
    # ------------------------------------------------------------------

    def _extraction_failed(self, arxiv_id: str, reason: str) -> str:
        log(f"Extraction failed for {arxiv_id}: {reason}")
        Database.mark_text_extraction_failed(self.db, arxiv_id, reason)
        return "extraction_failed"

    def process_paper(self, paper: Paper) -> str:
        """
        Ciclo completo de um paper: PDF -> GROBID/limpeza/tokens -> JSON
        de seções -> remove o PDF.

        Aproveita um PDF já presente em disco (não baixa de novo). O PDF
        só é apagado depois que o JSON foi gravado *e* registrado no
        banco; se a extração falhar, o PDF fica para nova tentativa ou
        para OCR.

        Returns:
            "ok", "download_failed" ou "extraction_failed".

        Raises:
            GrobidUnavailableError: o servidor parou de responder. Não é
                falha do paper, então nada é registrado contra ele.
        """
        arxiv_id = paper.arxiv_id

        row = Database.get_processing(self.db, arxiv_id)

        if row is not None and row["pdf_path"] and Path(row["pdf_path"]).is_file():
            pdf_path = Path(row["pdf_path"])
        else:
            pdf_path = self.download_pdf(paper)
            if not pdf_path.is_file():
                return "download_failed"

        try:
            document = self.processor.extract(pdf_path, arxiv_id)
        except Exception as e:
            # O extrator levanta o mesmo erro para "servidor fora do ar" e
            # para "documento problemático". Na dúvida, pergunta ao
            # servidor: se ele não responde, a culpa não é do paper.
            if not self.grobid.is_alive():
                raise GrobidUnavailableError(
                    f"GROBID stopped answering while processing {arxiv_id}: {e}"
                ) from e

            return self._extraction_failed(arxiv_id, f"{type(e).__name__}: {e}")

        try:
            sections_path, n_chars, n_tokens = self._preprocess_and_save(document)
        except PipelineError as e:
            return self._extraction_failed(arxiv_id, str(e))

        Database.mark_text_extracted(
            self.db,
            arxiv_id,
            sections_path,
            tei_path=document.tei_path,
            char_count=n_chars,
            token_count=n_tokens,
            section_count=len(document.sections),
            reference_count=len(document.references),
        )

        if not self.keep_pdfs:
            self._remove_pdf(arxiv_id, pdf_path)

        return "ok"

    def _preprocess_and_save(self, document: ExtractedDocument) -> tuple[Path, int, int]:
        """
        Limpa e tokeniza as seções e grava o JSON.

        O JSON só é gravado depois que o texto passa na checagem de
        tamanho, então um paper sem texto útil (PDF escaneado) não deixa
        arquivo no disco e, no reprocessamento, não sobrescreve o JSON
        bom que já existia.

        Returns:
            (caminho do JSON, nº de caracteres, nº de tokens)

        Raises:
            PipelineError: a limpeza falhou, não sobrou texto suficiente
                ou o JSON não pôde ser gravado.
        """
        try:
            self.processor.preprocess(document)
        except Exception as e:
            raise PipelineError(f"preprocessing failed: {type(e).__name__}: {e}") from e

        sections = list(document.sections)
        n_chars = sum(len(s.processed_text or "") for s in sections)
        n_tokens = sum(len(s.tokens or []) for s in sections)

        if n_chars < MIN_TEXT_CHARS:
            raise PipelineError(
                f"only {n_chars} characters of text after cleaning "
                f"({len(sections)} section(s)); the PDF may be scanned and need OCR"
            )

        try:
            sections_path = self.processor.save(document)
        except Exception as e:
            raise PipelineError(f"saving sections failed: {type(e).__name__}: {e}") from e

        return sections_path, n_chars, n_tokens

    def reprocess(self, limit: Optional[int] = None) -> dict[str, int]:
        """
        Refaz limpeza e tokenização a partir dos TEI já salvos, sem PDF e
        sem GROBID.

        Use depois de mudar o limpador (ex.: `remove_math=False`), as
        stopwords ou a lematização. Reescreve o JSON de seções e as
        contagens no banco.

        Um paper que não passa mais na checagem de tamanho (limpador
        configurado de forma agressiva demais) conserva o JSON anterior
        e é contado em `failed`.

        Returns:
            {"ok": ..., "missing_tei": ..., "failed": ...}
        """
        ids = Database.get_reprocessable(self.db, limit)
        stats = {"ok": 0, "missing_tei": 0, "failed": 0}
        log(f"{len(ids)} paper(s) to reprocess")

        for i, arxiv_id in enumerate(ids, start=1):
            row = Database.get_processing(self.db, arxiv_id)
            tei_path = Path(row["tei_path"])

            if not tei_path.is_file():
                log(f"[{i}/{len(ids)}] `{arxiv_id}`: TEI not found at {tei_path}; skipped")
                stats["missing_tei"] += 1
                continue

            try:
                document = self.processor.extract_from_tei(arxiv_id, tei_path)
                sections_path, n_chars, n_tokens = self._preprocess_and_save(document)
            except PipelineError as e:
                log(f"[{i}/{len(ids)}] `{arxiv_id}`: kept previous result ({e})")
                stats["failed"] += 1
                continue
            except Exception as e:
                log(f"[{i}/{len(ids)}] `{arxiv_id}`: could not read TEI ({type(e).__name__}: {e})")
                stats["failed"] += 1
                continue

            Database.mark_text_extracted(
                self.db,
                arxiv_id,
                sections_path,
                tei_path=tei_path,
                char_count=n_chars,
                token_count=n_tokens,
                section_count=len(document.sections),
                reference_count=len(document.references),
            )
            stats["ok"] += 1

        log(
            f"done: {stats['ok']} reprocessed, {stats['failed']} kept as before, "
            f"{stats['missing_tei']} without TEI"
        )
        return stats

    def prune_pdfs(self, dry_run: bool = False) -> dict[str, int]:
        """
        Apaga os PDFs de papers cujo texto já foi extraído, como os que
        sobram de execuções com `--keep-pdfs`.

        O TEI é conferido antes de cada remoção: um PDF nunca é apagado se
        puder ser a única cópia dos dados brutos. PDFs de papers sem texto
        (extração falhou) são mantidos, para nova tentativa ou OCR.

        Args:
            dry_run: só conta o que seria apagado.

        Returns:
            {"removed": ..., "kept_no_tei": ..., "already_gone": ..., "bytes": ...}
            (com `dry_run`, "removed" e "bytes" são o que seria apagado).
        """
        stats = {"removed": 0, "kept_no_tei": 0, "already_gone": 0, "bytes": 0}

        for arxiv_id, pdf_path in Database.get_prunable_pdfs(self.db):
            tei_path = Path(Database.get_processing(self.db, arxiv_id)["tei_path"])

            if not tei_path.is_file():
                stats["kept_no_tei"] += 1
                continue

            path = Path(pdf_path)

            if not path.is_file():
                stats["already_gone"] += 1
                if not dry_run:
                    Database.mark_pdf_removed(self.db, arxiv_id)
                continue

            size = path.stat().st_size

            if dry_run or self._remove_pdf(arxiv_id, path):
                stats["removed"] += 1
                stats["bytes"] += size

        log(
            f"{'would remove' if dry_run else 'removed'} {stats['removed']} PDF(s) "
            f"({stats['bytes'] / 1e6:.1f} MB); {stats['kept_no_tei']} kept because "
            f"their TEI is missing; {stats['already_gone']} were already gone"
        )
        return stats

    def process_pending(self, limit: Optional[int] = None) -> dict[str, int]:
        """
        Processa a fila de trabalho do banco: para cada paper sem
        texto, baixa o PDF (se preciso), extrai e pré-processa o texto,
        grava o JSON de seções e remove o PDF.

        Os que já têm PDF em disco vêm primeiro, pois não esperam o
        intervalo entre downloads.

        O lote para sozinho se o GROBID cair ou se houver
        `max_consecutive_failures` falhas seguidas.

        Args:
            limit: máximo de papers nesta execução (None = todos).

        Raises:
            PipelineError: o GROBID não está acessível ao começar.
        """
        if not self.grobid.is_alive():
            raise PipelineError(
                f"GROBID is not reachable at {self.grobid.base_url}. "
                f"Start it (docker run --rm --init --ulimit core=0 -p 8070:8070 "
                f"grobid/grobid:0.9.1-crf) or set GROBID_URL."
            )

        Database.reconcile_pdfs(self.db)
        queue = Database.get_work_queue(
            self.db, limit=limit, max_attempts=self.max_attempts
        )

        stats = {"ok": 0, "download_failed": 0, "extraction_failed": 0, "aborted": 0}
        log(f"{len(queue)} paper(s) to process")

        consecutive_failures = 0

        for i, arxiv_id in enumerate(queue, start=1):
            paper = self._paper_from_id(arxiv_id)
            if paper is None:
                continue

            log(f"[{i}/{len(queue)}] `{arxiv_id}`")

            try:
                outcome = self.process_paper(paper)
            except GrobidUnavailableError as e:
                log(f"GROBID became unavailable, stopping: {e}")
                stats["aborted"] = len(queue) - i + 1
                break

            stats[outcome] += 1
            consecutive_failures = 0 if outcome == "ok" else consecutive_failures + 1

            if consecutive_failures >= self.max_consecutive_failures:
                log(
                    f"{consecutive_failures} consecutive failures; stopping so the "
                    f"attempts of the remaining papers are not burned. "
                    f"Check GROBID and the network, then run again."
                )
                stats["aborted"] = len(queue) - i
                break

        log(
            f"done: {stats['ok']} ok, {stats['download_failed']} download failed, "
            f"{stats['extraction_failed']} extraction failed"
            + (f", {stats['aborted']} not processed" if stats["aborted"] else "")
        )
        return stats

    def run(
        self,
        limit: Optional[int] = None,
        search_limit: Optional[int] = None,
        download_only: bool = False,
        reprocess: bool = False,
        prune_pdfs: bool = False,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        Args:
            limit:
                Máximo de papers a processar nesta execução.
            search_limit:
                Se informado, antes de processar busca no arXiv
                (`build_corpus(..., append=True)`) até o corpus ter
                este total de registros. É o limite *global* do corpus,
                não a quantidade de novos: com 4025 papers no banco,
                um valor menor ou igual a 4025 não busca nada.
            download_only:
                Só baixa os PDFs pendentes; não chama o GROBID.
            reprocess:
                Só refaz limpeza e tokens a partir dos TEI salvos; não
                usa PDF nem GROBID (nem a API do arXiv).
            prune_pdfs:
                Só apaga os PDFs de papers que já têm texto e TEI.
            dry_run:
                Com `prune_pdfs`, só mostra o que seria apagado.
        """
        if prune_pdfs:
            return self.prune_pdfs(dry_run=dry_run)

        if reprocess:
            return self.reprocess(limit)

        if search_limit is not None:
            log(f"searching arXiv until the corpus has {search_limit} records")
            self.extractor.build_corpus(search_limit, append=True)

        if download_only:
            return self.download_pending(limit)

        return self.process_pending(limit)



    # def download(self, paper: Paper) -> tuple[str, Paper]:
    #     """Tries to download the Paper's PDF. Returns the Paper object and the path to it's PDF."""
    #     # Verifies if the Paper's PDF is really pending for download
    #     if not paper.stage == ProcessingStage.DOWNLOAD_PENDING:
    #         raise PipelineError("tried to download a Paper\'s PDF file, but the PDF was already downloaded.")

    #     # sets the stage to download
    #     paper.stage: ProcessingStage = ProcessingStage.DOWNLOADING

    #     # tries to download using Downloader class
    #     try:
    #         pdf_path: str = self.pdf_downloader.download(paper)

    #     except Exception as e:
    #         pdf_path: str = ""

    #         # if it's not possible to download the Paper's PDF
    #         # mark the stage as an error, and raise the exception
    #         paper.stage: ProcessingStage = ProcessingStage.INVALID_DOWNLOAD

    #         raise PipelineError from e

    #     # if the PDF has been download, update the stage and
    #     # return the tuple (path, paper)
    #     paper.stage: ProcessingStage = ProcessingStage.PDF_AVAILABLE
    #     return (pdf_path, paper)


    # def extract(self, paper: Paper) -> tuple[ExtractedDocument, Paper]:
    #     """Tries to extract the text from a Paper's PDF.
    #     Returns a tuple containing a document, the extracted text + info, and the Paper."""



    # def process(self, paper: Paper) -> Paper:
    #     """Download a paper, extract its text, and return its updated model."""
    #     pdf_path = self.pdf_downloader.download(paper)

    #     self.text_extractor.extract(
    #         paper,
    #         pdf_path=pdf_path,
    #     )

    #     return paper

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Baixa PDFs do arXiv, extrai e pré-processa o texto com o GROBID e remove os PDFs."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="máximo de papers a processar nesta execução")
    parser.add_argument("--search", type=int, default=None, metavar="TOTAL",
                        help="antes de processar, busca no arXiv até o corpus ter TOTAL registros")
    parser.add_argument("--download-only", action="store_true",
                        help="só baixa os PDFs pendentes; não chama o GROBID")
    parser.add_argument("--reprocess", action="store_true",
                        help="refaz limpeza e tokens a partir dos TEI salvos; não usa PDF nem GROBID")
    parser.add_argument("--prune-pdfs", action="store_true",
                        help="apaga os PDFs de papers que já têm texto e TEI (ex.: sobras de --keep-pdfs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="com --prune-pdfs, só mostra o que seria apagado")
    parser.add_argument("--keep-pdfs", action="store_true",
                        help="não apaga o PDF depois de extrair o texto")
    parser.add_argument("--grobid-url", default=None,
                        help="endereço do GROBID (padrão: $GROBID_URL ou http://localhost:8070)")
    parser.add_argument("--spacy-model", default=None,
                        help="modelo spaCy para lematização (ex.: en_core_web_sm); padrão: sem lematizar")
    args = parser.parse_args(argv)

    with PaperPipeline(
        keep_pdfs=args.keep_pdfs,
        spacy_model=args.spacy_model,
        grobid=GrobidClient(base_url=args.grobid_url) if args.grobid_url else None,
    ) as pipeline:
        pipeline.run(
            limit=args.limit,
            search_limit=args.search,
            download_only=args.download_only,
            reprocess=args.reprocess,
            prune_pdfs=args.prune_pdfs,
            dry_run=args.dry_run,
        )

if __name__ == "__main__":
    main()
