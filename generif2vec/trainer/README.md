This module contains scripts to train and evaluate the Doc2Vec model. The module
works with Google Cloud Platform's ai platform. 

For example (change `--package-path` for your machine and the `--abstracts_file` to your google storage bucket):
```shell script
gcloud ai-platform jobs submit training GENERIF2VEC_MODEL \
--scale-tier=CUSTOM \
--master-machine-type=n1-highmem-4 \
--job-dir=gs://generif2vec-mlengine \
--package-path=/Users/kevin/projects/generif2vec/generif2vec \
--module-name=generif2vec.trainer.task \
--python-version=3.5 \
--runtime-version=1.14 \
-- \
--abstracts_file=gs://generif2vec-mlengine/data/generif.pubmed.abstracts.df.txt.gz \
--epochs=20 \
--top_k=5
```