
"""
Relatório de estado do corpus: quantos artigos já conhecemos, quantos
PDFs já foram baixados, quantos textos já foram extraídos, e a
distribuição por categoria x intervalo de submissão.
 
Lê diretamente do banco SQLite (`src.Storage.Database`), sincronizado
pelo `CorpusBuilder` a cada coleta. Não refaz nenhuma consulta à API
do arXiv nem ao manifesto JSONL.
"""
from pathlib import Path
 
from src.Pipeline.Extraction import CorpusBuilder
from src.Storage import Database
 
 
def print_report(db_path: str | Path = Database.DEFAULT_DB_PATH) -> None:
    """Imprime o relatório de estado do corpus no console."""
    conn = Database.get_connection(db_path)
    Database.init_db(conn)
 
    try:
        report = Database.build_report(
            conn,
            date_ranges=CorpusBuilder.DATE_RANGES,
        )
    finally:
        conn.close()
 
    print(report)
 
 
def main() -> None:
    print_report()
 
 
if __name__ == "__main__":
    main()