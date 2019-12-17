import hypertune
import os
import umap
import warnings

import pandas as pd
import numpy as np

from generif2vec.text.util import process_abstracts_file
from generif2vec.text.util import tag_documents
from gensim.models.doc2vec import Doc2Vec
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import percentileofscore


def vec_for_learning(model, tagged_docs):
    """Make vectors suitable for downstream ML tasks"""
    logger.info("Inferring Vectors...")
    sents = tagged_docs
    targets, regressors = zip(
        *[
            (document.tags[0], model.infer_vector(document.words, steps=20))
            for document in sents
        ]
    )
    return np.array(targets), np.array(regressors)


def train_doc2vec(abstracts_file, dm=0, epochs=20, negative=5, min_count=5, hs=0, sample=0):
    """
    Train a doc2vec model on the abstracts file
    :param abstracts_file:
    :param dm:
    :param epochs:
    :param negative:
    :param min_count:
    :param hs:
    :param sample:
    :return:
    """
    processed_texts, labels = process_abstracts_file(abstracts_file, nrows=None)

    # split the documents into training and test sets
    train_docs, test_docs, train_labels, test_labels = train_test_split(
        processed_texts, labels, test_size=0.25, stratify=labels,
    )

    train_tagged = tag_documents(train_docs, train_labels)
    test_tagged = tag_documents(test_docs, test_labels)
    logger.info("Tagged {} training documents from {} genes".format(len(train_tagged), len(set(train_labels))))
    logger.info("Tagged {} test documents from {} genes".format(len(test_tagged), len(set(test_labels))))

    # initialize the doc2vec model
    model = Doc2Vec(
        dm=dm, vector_size=100, negative=negative, hs=hs, min_count=min_count, sample=sample, workers=4, epochs=epochs
    )

    # build the vocabulary
    model.build_vocab([x for x in train_tagged])

    # train the model for epochs iterations
    logger.info("Training Doc2Vec...")
    model.train(train_tagged, total_examples=len(train_tagged), epochs=model.epochs)

    # convert to arrays
    y_train, X_train = vec_for_learning(model, train_tagged)
    y_test, X_test = vec_for_learning(model, test_tagged)

    return model, X_train, X_test, y_train, y_test


def fit_umap(X_train_vec, X_test_vec, umap_components):
    """
    Reduce the dimensionality of training and test sets with UMAP
    :param X_train_vec: Training vectors from doc2vec
    :param X_test_vec: Test vectors from doc2vec
    :param umap_components: The number of components to reduce the dimensionality to.
    :return:
    """
    logger.info("Fitting UMAP...")
    # Embed the vectors using UMAP
    reducer = umap.UMAP(
        n_components=umap_components,
        min_dist=0.0,
        n_neighbors=30,
        metric='cosine',
    )
    X_train_embed = reducer.fit_transform(X_train_vec)
    X_test_embed = reducer.transform(X_test_vec)

    return X_train_embed, X_test_embed


def fit_logreg(X_train_embed, y_train):
    """
    Fit a logistic regression model to predict the most similar gene (umap embedded d2v vector)
    This serves
    :param X_train_embed: The training umap-embedded d2v vector
    :param y_train: The training labels
    :return: classifier object
    """
    logger.info("Fitting Logistic Regression classifier...")
    clf = LogisticRegression(
        multi_class='ovr',
        n_jobs=16,
        solver='saga',
        max_iter=200,
    )
    clf.fit(X_train_embed, y_train)
    return clf


def evaluate_rank(model, X_test_vec, y_test, top_k=5):
    """
    evaluate the doc2vec embeddings with performance of unseen data.
    :param model: A trained doc2vec model
    :param X_test_vec: doc2vec embedded test vectors
    :param y_test: test labels
    :return: top_k_accuracy, median gene rank, median difference between top similarity and gene similarity
    """
    # predict on the test set
    logger.info("Evaluating using doc2vec most_similar...")

    n_classes = len(np.unique(y_test))
    gene_rank = []
    similarity_loss = []
    in_top_k = []
    for i, gene in enumerate(y_test):
        try:
            similar_genes = model.docvecs.most_similar([X_test_vec[i, :]], topn=n_classes)
            similarity = pd.DataFrame(similar_genes, columns=['gene', 'score'])
            gene_similarity = similarity.loc[similarity['gene'] == gene]
            rank = (gene_similarity.index + 1)[0]
            gene_rank.append(rank)
            in_top_k.append(rank <= top_k)
            similarity_difference = similarity.iloc[0]['score'] - gene_similarity['score']
            similarity_loss.append(similarity_difference.values[0])
        except IndexError:
            # empty DataFrame
            continue

    top_k_accuracy = sum(in_top_k) / len(in_top_k)

    return top_k_accuracy, np.median(gene_rank), np.median(similarity_loss)


def train_and_evaluate(abstracts_file, outdir, dm=0, epochs=20, negative=15, min_count=2, hs=0, sample=0, top_k=10):
    """
    Train and evaluate Doc2vec model.
    :param abstracts_file: <str> Full path to the abstracts file output by download-abstracts.
    :param outdir: <str> Full path to the output directory where you want the models saved.
    :param dm:
    :param epochs: <int> Number of epochs to train the Doc2Vec model
    :param negative:
    :param min_count:
    :param hs:
    :param sample:
    :param top_k: <int> Number of class labels to allow for
    :return:  None, This functions saves models in outdir.
    """
    model, X_train, X_test, y_train, y_test = train_doc2vec(
        abstracts_file,
        dm=dm,
        epochs=epochs,
        negative=negative,
        min_count=min_count,
        hs=hs,
        sample=sample
    )

    top_k_accuracy, median_rank, median_sim_diff = evaluate_rank(model, X_test, y_test, top_k=top_k)

    # Calling the hypertune library and setting our metric
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='top_k_accuracy',
        metric_value=top_k_accuracy,
    )
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='median_rank',
        metric_value=median_rank,
    )
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='median_similiarity_difference',
        metric_value=median_sim_diff,
    )
    logger.info("Top {} accuracy: {}".format(top_k, top_k_accuracy))
    logger.info("Median gene rank: {}".format(median_rank))
    logger.info("Median Similarity Difference: {}".format(median_sim_diff))

    model_filename = 'generif2vec.doc2vec'
    try:
        model.save(os.path.join(outdir, model_filename))
    except (PermissionError, FileNotFoundError) as e:
        logger.warning("Could not write to {}.\n{}".format(outdir, e))


def evaluate(clf, X_test_embedded, y_test, top_k=5):
    """
    Since the labels are sparse, evaluate the doc2vec embeddings with performance of logistic regression using top_k.
    :param clf:
    :param X_test_embedded:
    :param y_test:
    :param top_k:
    :return: probs, top_k_accuracy, mean gene ranks
    """
    # predict on the test set
    logger.info("Evaluating...")
    probs = clf.predict_proba(X_test_embedded)
    best_n = np.argsort(probs, axis=1)[:, -top_k:]

    in_top_k = []
    gene_ranks = []
    for i, gene in enumerate(y_test):
        if gene in best_n[i, :]:
            in_top_k.append(True)
        else:
            in_top_k.append(False)
        #
        gene_ranks.append(percentileofscore(probs[i, :], probs[i, gene]))

    top_k_accuracy = sum(in_top_k) / len(in_top_k)

    return probs, top_k_accuracy, np.mean(gene_ranks)


def train_and_evaluate_logreg(abstracts_file, outdir, epochs, umap_components, top_k):
    """
    Train and store Doc2Vec, UMAP, and logistic regression models
    :param abstracts_file: <str> Full path to the abstracts file output by download-abstracts.
    :param epochs: <int> Number of epochs to train the Doc2Vec model
    :param outdir: <str> Full path to the output directory where you want the models saved.
    :param umap_components: <int> Number of components for the UMAP dimensionality reduction.
    :param top_k: <int> Number of class labels to allow for
    :return: None, This functions saves models in outdir.
    """
    model, X_train, X_test, y_train, y_test = train_doc2vec(abstracts_file, epochs, outdir)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train_embed, X_test_embed = fit_umap(X_train, X_test, umap_components)
    del X_train
    del X_test

    clf = fit_logreg(X_train_embed, y_train)
    probs, top_k_accuracy, gene_ranks = evaluate(clf, X_test_embed, y_test, top_k=top_k)

    # Calling the hypertune library and setting our metric
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='top_k_accuracy',
        metric_value=top_k_accuracy,
    )
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mean_percentile_rank',
        metric_value=gene_ranks,
    )
    logger.info("Top {} accuracy: {}".format(top_k, top_k_accuracy))
    logger.info("Mean percentile Rank: {}".format(gene_ranks))
