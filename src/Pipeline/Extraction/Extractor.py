"""src.Pipeline.Extraction.Extractor
Construção e persistência de um corpus de artigos científicos do arXiv.

Responsabilidades:
    - Consultar a API do arXiv por categoria e intervalo de datas.
    - Normalizar os metadados dos artigos.
    - Deduplicar registros pelo identificador arXiv.
    - Construir ou atualizar um corpus com limite global.
    - Persistir o manifesto JSONL e seus metadados.
    - Converter registros para objetos Paper.

A extração dos PDFs não pertence a este módulo. Ela deve ser realizada
pelo pipeline de processamento científico do projeto.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

import arxiv  # type: ignore[import-not-found]

from src.Config import config
from src.Models import Paper
from src.Storage import Database


logger = logging.getLogger(__name__)

def log(message: str) -> None:
    print(message)

ResultCorpus: TypeAlias = list[arxiv.Result]
CorpusRecord: TypeAlias = dict[str, Any]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "corpus" / "manifest.jsonl"
)

DEFAULT_METADATA_PATH = (
    PROJECT_ROOT / "data" / "corpus" / "metadata.json"
)


# ---------------------------------------------------------------------------
# Record utilities
# ---------------------------------------------------------------------------

def read_record(path: Path, index: int) -> CorpusRecord:
    """
    Lê um registro JSONL pelo índice, começando em zero.

    Raises:
        ValueError: índice negativo.
        IndexError: índice inexistente.
        json.JSONDecodeError: registro JSON inválido.
    """
    if index < 0:
        raise ValueError("index must be non-negative")

    with path.open("r", encoding="utf-8") as file:
        for current_index, line in enumerate(file):
            if current_index == index:
                return json.loads(line)

    raise IndexError(f"Record {index} not found in {path}")


def normalize_paper(result: arxiv.Result) -> CorpusRecord:
    """
    Converte um resultado da API do arXiv em um registro serializável.

    O identificador sem versão é utilizado para deduplicação.
    O identificador versionado é preservado para rastreabilidade.
    """
    versioned_id = result.get_short_id()
    arxiv_id = versioned_id.rsplit("v", 1)[0]

    return {
        "arxiv_id": arxiv_id,
        "versioned_arxiv_id": versioned_id,
        "title": " ".join(result.title.split()),
        "authors": [author.name for author in result.authors],
        "abstract": " ".join(result.summary.split()),
        "primary_subject": result.primary_category,
        "subjects": sorted(result.categories),
        "submission_date": result.published.isoformat(),
        "updated_date": result.updated.isoformat(),
        "doi": result.doi,
        "comment": result.comment,
        "pdf_url": result.pdf_url,
        "abstract_url": result.entry_id,
    }


def build_query(
    category: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Constrói uma consulta por categoria e intervalo de submissão.

    As datas devem estar no formato YYYYMMDD.
    """
    for name, value in (
        ("start_date", start_date),
        ("end_date", end_date),
    ):
        if len(value) != 8 or not value.isdigit():
            raise ValueError(
                f"{name} must use YYYYMMDD format: {value!r}"
            )

        datetime.strptime(value, "%Y%m%d")

    if start_date > end_date:
        raise ValueError(
            f"start_date {start_date} must not exceed end_date {end_date}"
        )

    if not category or not category.strip():
        raise ValueError("category must not be empty")

    return (
        f"cat:{category.strip()} "
        f"AND submittedDate:[{start_date}0000 TO {end_date}2359]"
    )


# ---------------------------------------------------------------------------
# Atomic persistence
# ---------------------------------------------------------------------------

def _atomic_write(
    path: Path,
    writer,
) -> None:
    """
    Escreve um arquivo temporário no mesmo diretório e substitui
    o destino após concluir a escrita.

    A substituição é atômica no sistema de arquivos, desde que
    origem e destino estejam no mesmo sistema de arquivos.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            writer(file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_json_atomic(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Persiste um objeto JSON sem deixar arquivos parcialmente escritos."""

    def writer(file) -> None:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    _atomic_write(path, writer)


def write_jsonl_atomic(
    path: Path,
    records: list[CorpusRecord],
) -> None:
    """Persiste uma coleção de registros JSONL atomicamente."""

    def writer(file) -> None:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    _atomic_write(path, writer)


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

class CorpusBuilder:
    """
    Constrói e mantém um corpus de metadados do arXiv.

    build_corpus(limit=100):
        Constrói um corpus novo com até 100 artigos únicos.

    build_corpus(limit=100, append=True):
        Atualiza o corpus existente, buscando artigos até que o
        total alcance 100 registros únicos.

    O limite é global, não por categoria ou intervalo de datas.

    Observação:
        O manifesto e o arquivo de metadados são substituídos
        atomicamente, cada um individualmente. A atualização dos
        dois arquivos não constitui uma transação única.
    """

    MANIFEST_PATH = DEFAULT_MANIFEST_PATH
    METADATA_PATH = DEFAULT_METADATA_PATH

    configured_categories = config["DEFAULT_CATEGORIES"]
    if isinstance(configured_categories, str):
        CATEGORIES: list[str] = [configured_categories]
    elif isinstance(configured_categories, Path):
        CATEGORIES = [str(configured_categories)]
    else:
        CATEGORIES = [str(category) for category in configured_categories]

    CATEGORY = str(config["DEFAULT_CATEGORY"])

    PAGE_SIZE = int(str(config["ARXIV_CLIENT_PAGE_SIZE"]))
    START_DATE = str(config["DEFAULT_START_DATE"])
    END_DATE = str(config["DEFAULT_END_DATE"])

    DELAY_SECONDS = float(
        str(config["ARXIV_CLIENT_DELAY_SECONDS"])
    )

    NUM_RETRIES = int(
        str(config["ARXIV_CLIENT_MAX_RETRIES"])
    )

    DATE_RANGES: list[tuple[str, str]] = [
        ("20000101", "20031231"),
        ("20040101", "20061231"),
        ("20070101", "20091231"),
        ("20100101", "20121231"),
        ("20130101", "20151231"),
        ("20160101", "20181231"),
        ("20190101", "20211231"),
        ("20220101", "20251231"),
    ]

    def __init__(
        self,
        path_to_manifest: Path = MANIFEST_PATH,
        path_to_metadata: Path = METADATA_PATH,
        category: str | None = None,
        page_size: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        db_path: Path | str | None = Database.DEFAULT_DB_PATH,
    ) -> None:
        self.path_manifest = Path(path_to_manifest)
        self.path_metadata = Path(path_to_metadata)

        # Caminho do banco SQLite usado para sincronizar o manifesto
        # a cada persistência. None desativa a sincronização.
        self.db_path = Path(db_path) if db_path is not None else None

        self.category = category or self.CATEGORY
        self.page_size = page_size or self.PAGE_SIZE

        self.start_date = start_date or self.START_DATE
        self.end_date = end_date or self.END_DATE

        if self.page_size <= 0:
            raise ValueError("page_size must be positive")

        self.query = build_query(
            category=self.category,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.client = arxiv.Client(
            page_size=self.page_size,
            delay_seconds=self.DELAY_SECONDS,
            num_retries=self.NUM_RETRIES,
        )

        # Estado da execução atual.
        self.examined = 0
        self.added = 0
        self.records: list[CorpusRecord] = []
        self.seen_ids: set[str] = set()

        # Mantido para compatibilidade com código que utilizava
        # o atributo original. Não representa PDFs extraídos.
        self.extracted = 0

        print(f"`CorpusBuild` initialized with parameters:")
        print(f"  category: {self.category}")
        print(f"  page_size: {self.page_size}")
        print(f"  start_date: {self.start_date}")
        print(f"  end_date: {self.end_date}")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def date_ranges(self) -> list[tuple[str, str]]:
        """Intervalos temporais investigados pelo projeto."""
        return list(self.DATE_RANGES)

    @property
    def categories(self) -> list[str]:
        """Categorias configuradas para a coleta."""
        configured = self.CATEGORIES

        if isinstance(configured, str):
            return [configured]

        return list(configured)

    # ------------------------------------------------------------------
    # Manifest access
    # ------------------------------------------------------------------

    def get_corpus(self, verbose: bool = False) -> list[CorpusRecord]:
        """
        Carrega o manifesto atual.

        Cada linha deve conter um objeto JSON válido com arxiv_id.
        Registros duplicados ou inválidos não são silenciosamente
        descartados: um manifesto inconsistente gera uma exceção.

        Args:
            verbose:
                Se True, imprime cada registro lido durante o
                carregamento. Desligado por padrão para evitar
                despejar milhares de linhas em manifestos grandes.
        """
        if not self.path_manifest.exists():
            return []

        records: list[CorpusRecord] = []
        seen_ids: set[str] = set()


        log(
            f"Carregando manifesto {self.path_manifest} "
            f"({self.path_manifest.stat().st_size} bytes)"
        )

        with self.path_manifest.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                    if verbose:
                        log(
                            f"[{line_number}] :: `{record}`"
                        )

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.path_manifest}, "
                        f"line {line_number}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected an object in "
                        f"{self.path_manifest}, line {line_number}"
                    )

                arxiv_id = record.get("arxiv_id")

                if not isinstance(arxiv_id, str) or not arxiv_id:
                    raise ValueError(
                        f"Missing arxiv_id in "
                        f"{self.path_manifest}, line {line_number}"
                    )

                if arxiv_id in seen_ids:
                    raise ValueError(
                        f"Duplicate arxiv_id {arxiv_id!r} "
                        f"in {self.path_manifest}"
                    )

                seen_ids.add(arxiv_id)
                records.append(record)

        return records

    @classmethod
    def record(
        cls,
        metadata: Path,
        index: int,
    ) -> CorpusRecord:
        """Recupera um registro pelo índice de um arquivo JSONL."""
        return read_record(Path(metadata), index)

    # ------------------------------------------------------------------
    # Domain conversion
    # ------------------------------------------------------------------

    @classmethod
    def from_record(
        cls,
        record: CorpusRecord,
        source_record_index: int,
        dataset_id: str = "arxiv-api",
        dataset_split: str = "train",
    ) -> Paper:
        """
        Converte um registro do manifesto em um objeto Paper.

        A construção do objeto de domínio é delegada ao modelo,
        evitando duplicar suas regras de validação neste módulo.
        """
        return Paper.from_source_record(
            record,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            source_record_index=source_record_index,
        )

    def to_papers(
        self,
        records: list[CorpusRecord] | None = None,
        dataset_id: str = "arxiv-api",
        dataset_split: str = "train",
    ) -> list[Paper]:
        """Converte vários registros para objetos Paper."""
        selected_records = (
            records if records is not None else self.get_corpus()
        )

        return [
            self.from_record(
                record=record,
                source_record_index=index,
                dataset_id=dataset_id,
                dataset_split=dataset_split,
            )
            for index, record in enumerate(selected_records)
        ]

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def try_add_to_corpus(
        self,
        results: list[arxiv.Result],
    ) -> None:
        """
        Normaliza e adiciona resultados inéditos ao corpus em memória.

        self.examined:
            Quantidade de resultados da API examinados.

        self.added:
            Quantidade de registros adicionados nesta execução.

        self.extracted:
            Mantido por compatibilidade; representa o número atual
            de registros em memória, não PDFs extraídos.
        """
        for result in results:
            self.examined += 1

            record = normalize_paper(result)
            arxiv_id = record["arxiv_id"]

            if arxiv_id in self.seen_ids:
                logger.debug(
                    "Ignoring duplicate arXiv record: %s",
                    arxiv_id,
                )
                continue

            self.seen_ids.add(arxiv_id)
            self.records.append(record)
            self.added += 1

        self.extracted = len(self.records)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(
        self,
        category: str,
        start_date: str,
        end_date: str,
        limit: int,
    ) -> list[arxiv.Result]:
        """Executa uma consulta limitada à categoria e ao intervalo."""
        query = build_query(
            category=category,
            start_date=start_date,
            end_date=end_date,
        )

        self.query = query

        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        return list(self.client.results(search))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, limit: int) -> None:
        """Persiste o manifesto e os metadados da coleta."""

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "arXiv API",
            "categories": self.categories,
            "date_ranges": [
                {
                    "start_date": start,
                    "end_date": end,
                }
                for start, end in self.date_ranges
            ],
            "requested_limit": limit,
            "record_count": len(self.records),
            "records_examined": self.examined,
            "records_added": self.added,
            "manifest": str(self.path_manifest),
        }

        # Primeiro, publica o manifesto completo.
        write_jsonl_atomic(
            self.path_manifest,
            self.records,
        )

        # Depois, publica os metadados correspondentes.
        write_json_atomic(
            self.path_metadata,
            metadata,
        )

        # Por fim, sincroniza o banco de consulta (se configurado).
        # Uma falha aqui não deve derrubar uma coleta já persistida
        # com sucesso no manifesto; ela é logada e seguida.
        if self.db_path is not None:
            try:
                conn = Database.get_connection(self.db_path)
                try:
                    Database.init_db(conn)
                    Database.upsert_many(conn, self.records)
                finally:
                    conn.close()
            except Exception:
                logger.exception(
                    "Failed to sync corpus to database at %s. "
                    "The manifest was persisted successfully.",
                    self.db_path,
                )

    # ------------------------------------------------------------------
    # Corpus construction
    # ------------------------------------------------------------------

    def build_corpus(
        self,
        limit: int = 100,
        append: bool = False,
        debug: bool = True,
    ) -> list[CorpusRecord]:
        """
        Constrói ou atualiza o corpus.

        Args:
            limit:
                Número máximo de registros únicos no corpus final.

            append:
                False: inicia um corpus novo em memória e substitui
                       o manifesto após uma coleta bem-sucedida.
                True: carrega o manifesto existente e adiciona
                      registros inéditos até atingir o limite.

            debug:
                Se True (padrão), imprime no console as mesmas
                mensagens enviadas ao logger, útil para execução
                interativa (ex.: blocos org-babel). Se False, o
                progresso só é reportado via `logger`.

        Returns:
            Lista de registros do corpus resultante.

        Raises:
            ValueError:
                Se o limite for negativo ou igual a zero.
            RuntimeError:
                Se uma construção nova não encontrar artigos.
            Exception:
                Falhas da API ou da persistência são propagadas.

        Importante:
            Uma falha durante a coleta não publica um manifesto
            parcialmente construído.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")

        def dlog(message: str) -> None:
            if debug:
                log(message)

        # Reinicia as estatísticas da execução atual.
        self.examined = 0
        self.added = 0

        if append:
            self.records = self.get_corpus()
        else:
            self.records = []

        self.seen_ids = {
            record["arxiv_id"]
            for record in self.records
        }

        self.extracted = len(self.records)

        if len(self.records) >= limit:
            logger.info(
                "Corpus already contains %d records; limit is %d.",
                len(self.records),
                limit,
            )

            dlog(
                f"Corpus already contains {len(self.records)} records; "
                f"limit is {limit}."
            )

            return self.records

        dlog(
            f"Building corpus: categories={len(self.categories)}, "
            f"date_ranges={len(self.date_ranges)}, limit={limit}"
        )

        categories = self.categories

        if not categories:
            raise ValueError("At least one category is required")

        logger.info(
            "Building corpus: categories=%d, date_ranges=%d, limit=%d",
            len(categories),
            len(self.date_ranges),
            limit,
        )

        try:
            for category in categories:
                for start_date, end_date in self.date_ranges:

                    # O limite é aplicado ao total de registros únicos.
                    remaining = limit - len(self.records)

                    if remaining <= 0:
                        break

                    logger.info(
                        "Searching category=%s, dates=%s..%s, "
                        "remaining=%d",
                        category,
                        start_date,
                        end_date,
                        remaining,
                    )

                    dlog(
                        f"Searching category={category}, "
                        f"dates={start_date}..{end_date}, "
                        f"remaining={remaining}"
                    )

                    results = self._search(
                        category=category,
                        start_date=start_date,
                        end_date=end_date,
                        limit=remaining,
                    )

                    self.try_add_to_corpus(results)

                    logger.info(
                        "Search returned %d results; "
                        "corpus now contains %d unique records.",
                        len(results),
                        len(self.records),
                    )

                    dlog(
                        f"Search returned {len(results)} results; "
                        f"corpus now contains {len(self.records)} unique records."
                    )

                if len(self.records) >= limit:
                    dlog(
                        f"Corpus built: total={len(self.records)}, "
                        f"examined={self.examined}, added={self.added}"
                    )
                    break

        except Exception:
            logger.exception(
                "Corpus collection failed. "
                "The previous manifest has not been replaced."
            )

            dlog(
                "Corpus collection failed. "
                "The previous manifest has not been replaced."
            )
            raise

        # Evita destruir um manifesto válido quando uma construção
        # nova não encontra nenhum resultado.
        if not self.records:
            raise RuntimeError(
                "Nenhum artigo correspondeu aos filtros. "
                "O manifesto anterior foi preservado."
            )

        # Proteção adicional: nunca persiste mais registros que o limite.
        self.records = self.records[:limit]
        self.seen_ids = {
            record["arxiv_id"]
            for record in self.records
        }
        self.extracted = len(self.records)

        self._persist(limit=limit)

        logger.info(
            "Corpus persisted: total=%d, examined=%d, added=%d",
            len(self.records),
            self.examined,
            self.added,
        )


        dlog(
            f"Corpus persisted: total={len(self.records)}, "
            f"examined={self.examined}, added={self.added}"
        )
        return list(self.records)

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill(
        self,
        date_ranges: list[tuple[str, str]],
        limit_per_range: int = 100,
        debug: bool = True,
    ) -> list[CorpusRecord]:
        """
        Busca novos artigos especificamente nos `date_ranges`
        informados, com uma cota própria (`limit_per_range`) para
        cada intervalo — independente do total já acumulado no
        corpus.

        Diferente de `build_corpus`, que para assim que o `limit`
        global é atingido (e por isso pode nunca alcançar os
        intervalos mais recentes, caso os mais antigos já preencham
        a cota sozinhos), este método é pensado para preencher
        lacunas conhecidas de cobertura temporal: cada intervalo
        recebe sua própria chance de busca, sempre.

        Use `Database.get_empty_date_ranges` para descobrir quais
        intervalos estão sem nenhum artigo antes de chamar este
        método.

        Reaproveita a mesma busca, deduplicação (por `seen_ids`) e
        persistência de `build_corpus` — só muda a estratégia de
        iteração. Sempre opera em modo "append": carrega o
        manifesto existente e adiciona o que for inédito.

        Args:
            date_ranges:
                Intervalos a investigar, no formato de
                `CorpusBuilder.DATE_RANGES` (strings 'YYYYMMDD').
            limit_per_range:
                Máximo de registros únicos novos a buscar por
                intervalo (não por categoria x intervalo).
            debug:
                Mesma semântica de `build_corpus`.

        Returns:
            Lista completa do corpus resultante (artigos antigos +
            os eventualmente encontrados nos intervalos informados).

        Raises:
            ValueError:
                Se `limit_per_range` não for positivo, ou se
                `date_ranges` estiver vazio.
        """
        if limit_per_range <= 0:
            raise ValueError("limit_per_range must be positive")

        if not date_ranges:
            raise ValueError("date_ranges must not be empty")

        def dlog(message: str) -> None:
            if debug:
                log(message)

        self.examined = 0
        self.added = 0

        self.records = self.get_corpus()
        self.seen_ids = {
            record["arxiv_id"]
            for record in self.records
        }
        self.extracted = len(self.records)

        categories = self.categories

        if not categories:
            raise ValueError("At least one category is required")

        dlog(
            f"Backfilling {len(date_ranges)} date range(s) "
            f"across {len(categories)} categor(y/ies), "
            f"limit_per_range={limit_per_range}"
        )

        try:
            for category in categories:
                for start_date, end_date in date_ranges:
                    dlog(
                        f"Backfill: category={category}, "
                        f"dates={start_date}..{end_date}, "
                        f"limit_per_range={limit_per_range}"
                    )

                    results = self._search(
                        category=category,
                        start_date=start_date,
                        end_date=end_date,
                        limit=limit_per_range,
                    )

                    before = len(self.records)
                    self.try_add_to_corpus(results)
                    added_here = len(self.records) - before

                    dlog(
                        f"Search returned {len(results)} results, "
                        f"{added_here} new; "
                        f"corpus now contains {len(self.records)} unique records."
                    )

        except Exception:
            logger.exception(
                "Backfill failed. The previous manifest has not been replaced."
            )
            dlog(
                "Backfill failed. The previous manifest has not been replaced."
            )
            raise

        self.extracted = len(self.records)

        # requested_limit nos metadados é só uma estimativa de teto
        # para este modo (cota por intervalo x intervalos x categorias),
        # não um limite global de fato aplicado.
        estimated_ceiling = limit_per_range * len(date_ranges) * len(categories)
        self._persist(limit=max(estimated_ceiling, len(self.records)))

        dlog(
            f"Backfill persisted: total={len(self.records)}, "
            f"examined={self.examined}, added={self.added}"
        )

        return list(self.records)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def to_paper(
    record: CorpusRecord,
    source_record_index: int = 0,
    dataset_id: str = "arxiv-api",
    dataset_split: str = "train",
) -> Paper:
    """
    Converte um registro individual em Paper.

    Mantém uma função de módulo para código que utilizava to_paper()
    antes da refatoração.
    """
    return CorpusBuilder.from_record(
        record=record,
        source_record_index=source_record_index,
        dataset_id=dataset_id,
        dataset_split=dataset_split,
    )