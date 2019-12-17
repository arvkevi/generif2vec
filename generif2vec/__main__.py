import fire
import glob
import os
import sys

from loguru import logger
from generif2vec.text.util import tokenize
from generif2vec.trainer.task import main as train_models
from gensim.models.doc2vec import Doc2Vec


def similar_genes(model_file, texts_dir, n_similar=10, outdir=None):
    """
    Get the top n_similar genes to the descriptions provided in the text files
    :param model_file: Path the doc2vec binary model
    :param texts_dir: Path to directory with text files.
    :param n_similar: <int> Number of similar genes to output.
    :param outdir: <path to output>
    :return:
    """
    try:
        model = Doc2Vec.load(model_file)
    except (FileExistsError, FileNotFoundError):
        logger.critical("Please provide a valid model file")
        sys.exit(1)

    texts = []
    for text_file in list(glob.glob('*.txt')):
        try:
            with open(text_file, 'r') as f:
                texts.append((f.readlines(), os.path.basename(text_file).strip('.txt')))
        except (FileNotFoundError, PermissionError):
            logger.critical("Please provide a valid text file")
            sys.exit(1)

    processed_texts = tokenize([text for text, filename in texts])
    vectors = [model.infer_vector([tokens]) for tokens in processed_texts]
    results = [(model.docvecs.most_similar(vector, topn=n_similar), filename) for vector, (text, filename) in zip(vectors, texts)]

    if outdir is None:
        outdir = os.getcwd()

    with open(os.path.join(outdir, 'generif2vec.{}_similar.results'.format(n_similar)), 'w') as f:
        try:
            for result in results:
                for gene, similarity, filename in result:
                    f.write('{}\t{}\t{}\n'.format(gene, similarity, filename))
        except (FileNotFoundError, PermissionError):
            logger.critical("Please provide a valid output directory")
            sys.exit(1)


def main():
    fire.Fire(
        {"train-models": train_models,
         "similarity": similar_genes,
         }
    )


if __name__ == "__main__":
    main()
