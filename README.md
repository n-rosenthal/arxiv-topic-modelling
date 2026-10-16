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