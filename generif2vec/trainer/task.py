import argparse
import sys

from generif2vec.config import DATA_DIRECTORY
from generif2vec.trainer import model
from loguru import logger


def main(abstracts_file, job_dir, dm=0, epochs=20, negative=15, min_count=2, hs=0, sample=0, top_k=10):
    """
    Train and evaluate the models.
    :param abstracts_file: Path to the geneRIF abstracts file
    :param job_dir: The directory to output the models
    :param dm: PV-DM (1) or PV-DBOW (0)
    :param epochs: Number of training epochs for doc2vec
    :param negative: negative sampling (noise words to draw)
    :param min_count: Ignore words with frequency less than this.
    :param hs: Hierarchical softmax (1) or not (0)
    :param sample: Threshold for which higher frequency words are downsampled.
    :param top_k: Number of labels allowed to be predicted before the truth label is counted as correct.
    :return:
    """
    if abstracts_file is not None:
        abstracts_file = abstracts_file
    else:
        logger.critical("You must provide a path to the zipped generif abstracts file.")
        sys.exit(1)

    if job_dir is not None:
        outdir = job_dir
    else:
        outdir = DATA_DIRECTORY

    if epochs is not None:
        epochs = int(epochs)

    if top_k is not None:
        top_k = int(top_k)

    model.train_and_evaluate(
        abstracts_file=abstracts_file,
        outdir=outdir,
        dm=dm,
        epochs=epochs,
        negative=negative,
        min_count=min_count,
        hs=hs,
        sample=sample,
        top_k=top_k
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--abstracts_file',
        help='Full path to the abstracts file output by download-abstracts.',
        type=str,
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='Full path to the output directory where you want the models saved.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--dm',
        help='Use Distributed Memory if dm=1 else use bag-of-words if dm=0',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--epochs',
        help='Number of epochs to train the Doc2Vec model.',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--negative',
        help='How many noise words',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--min_count',
        help='Ignores all words with total frequency lower than this.',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--hs',
        help='Use Hierarchical softmax when hs=1',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--sample',
        help='downsample frequency',
        type=float,
        default=0,
    )
    parser.add_argument(
        '--top_k',
        help='Number of class labels to allow an predicted class to be identified as "correct',
        type=int,
        default=5,
    )

    args = parser.parse_args()
    main(args.abstracts_file, args.job_dir, args.dm, args.epochs, args.negative, args.min_count, args.hs, args.sample, args.top_k)
