# arxiv-topic-modelling

## start GROBID
Run the GROBID service in Docker in a separate terminal:

```bash
docker run --rm \
    --name grobid \
    -p 8070:8070 \
    lfoppiano/grobid:0.8.2
```

we can check its health point in

```bash
curl -f http://localhost:8070/api/isalive
```

this is necessary only for the preprocessing of PDF to plain text data.

## Downloading pendent PDFs

```bash
python -m src.Pipeline.pipeline --download-only --limit 500
``` 

## PDF to text extraction with GROBID
```
# 0. backup: o banco ganha 5 colunas novas na primeira execução (migração automática)
cp data/corpus/papers.db data/corpus/papers.db.bak

# 1. subir o GROBID (imagem CRF: ~500 MB, sem GPU)
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.1-crf

# 2. teste pequeno, depois inspecione data/texts/*.txt e *.tei.xml
python -m src.Pipeline.pipeline --limit 3

# 3. lote
python -m src.Pipeline.pipeline --limit 500
```