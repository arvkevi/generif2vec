import gcsfs
import pandas as pd
import spacy

from gensim.models.doc2vec import TaggedDocument
from loguru import logger


def process_abstracts_file(abstracts_file, nrows=None):
    """
    Process an abstracts file output by text.download_abstracts.download_and_process_abstracts
    :param abstracts_file: The
    :param nrows: Limit the number of records to read from the abstracts file
    :param skiprows: skip these rows (use numpy.random.choice)
    :return:
    """
    data = pd.read_csv(abstracts_file, sep="\t", nrows=nrows)
    data.drop(columns=["Unnamed: 0"], inplace=True)
    df = data.drop_duplicates(['pmid', 'gene_symbol'])
    df = df.reset_index(drop=True)

    # limit to genes with at least 10 abstracts describing them.
    df = df[df.groupby("gene_symbol")["gene_symbol"].transform("size") > 10]

    raw_texts = df["abstract"].tolist()
    processed_texts = tokenize(raw_texts)

    labels = df["gene_symbol"].tolist()
    logger.info("Processed {} texts from {} unique genes".format(len(processed_texts), df['gene_symbol'].nunique()))

    return processed_texts, labels


def tokenize(raw_abstracts):
    """
    tokenize the text with spacy and remove stopwords
    :param raw_abstracts: List of texts
    :return:
    """
    # use spacy to process the raw text into spcay documents
    try:
        nlp = spacy.load("en")
    except OSError:
        from spacy.cli import download

        download("en")
        nlp = spacy.load("en")

    texts = []
    for doc in nlp.pipe(
        raw_abstracts, n_threads=4, disable=["tagger", "parser", "ner"]
    ):
        texts.append(
            [preprocess_token(token) for token in doc if is_token_allowed(token)]
        )
    return texts


def tag_documents(processed_texts, labels):
    """
    Create Tagged Documents for doc2vec
    :param processed_texts: Processed texts from process_abstracts_file
    :param labels: labels from process_abstracts_file
    :return: tagged documents for training doc2vec
    """
    tagged_docs = [
        TaggedDocument(doc, [label]) for doc, label in zip(processed_texts, labels)
    ]

    return tagged_docs


def is_token_allowed(token):
    """
        Only allow valid tokens which are not stop words
        and punctuation symbols.
    """
    if not token or not token.string.strip() or token.is_stop or token.is_punct:
        return False
    return True


def preprocess_token(token):
    """Reduce token to its lowercase lemma form"""
    return token.lemma_.strip().lower()
