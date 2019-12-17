import gzip
import time
import urllib.request

import pandas as pd

from Bio import Entrez
from biothings_client import get_client
from generif2vec.config import DATA_DIRECTORY, ENTREZ_USER_NAME, ENTREZ_API_KEY


def download_generifs(data_directory):
    """download the generif annotations"""
    generifs_basic = 'ftp://ftp.ncbi.nih.gov/gene/GeneRIF/generifs_basic.gz'
    urllib.request.urlretrieve(generifs_basic, '{}/generifs_basic.gz'.format(DATA_DIRECTORY))


def process_generifs():
    """process the generifs annotations file"""
    genes = []
    pubmed_ids = []

    human_tax_id = '9606'
    with gzip.open('{}/generifs_basic.gz'.format(DATA_DIRECTORY), 'rb') as f:
        for line in f:
            # only want genes annotated in humans
            if line.startswith(human_tax_id):
                tax_id, gene_id, pubmed_id, timestamp, text = line.strip().split('\t')
                genes.append(gene_id)
                pubmed_ids.append(pubmed_id)

    # write file containing the list of unique pubmed ids
    with open('generif.uniq.pmids', 'w') as f:
        for pmid in set(pubmed_ids):
            f.write(f'{pmid}\n')

    pubmed_ids = list(set(pubmed_ids))
    return pubmed_ids, genes


def download_abstracts(pubmed_ids):
    """generic function to download abstracts and dump to file"""
    Entrez.email = ENTREZ_USER_NAME
    Entrez.api_key = ENTREZ_API_KEY

    counter = 0
    step = 10000
    no_abstract = []
    for i in range(step, len(pubmed_ids), step):
        pubmed_ids_subset = pubmed_ids[counter:i]
        handle = Entrez.efetch(db='pubmed',
                               id=','.join(map(str, pubmed_ids_subset)),
                               rettype="xml",
                               retmode="text",
                               retmax=f"{step}",
                               )

        records = Entrez.read(handle)

        for pubmed_article in records['PubmedArticle']:
            pmid = int(str(pubmed_article['MedlineCitation']['PMID']))
            article = pubmed_article['MedlineCitation']['Article']
            if 'Abstract' in article:
                with open('{}/generif.abstracts.txt'.format(DATA_DIRECTORY), 'a') as f:
                    f.write(f"{pmid}\t{article['Abstract']['AbstractText'][0]}\n")
            else:
                no_abstract.append(pmid)

        counter += step
        time.sleep(1)


def remove_gene_names(row):
    gene_symbols = []
    if row['gene_symbol']:
        gene_symbols.append(row['gene_symbol'])
    if row['gene_alias']:
        gene_symbols.extend(row['gene_alias'])
    abstract = row['abstract']

    if gene_symbols:
        for symbol in gene_symbols:
            abstract = abstract.replace(symbol, '$GENE@')
    return abstract


def download_and_process_abstracts():
    """Download and process the abstracts from GeneRIF human annotations"""
    download_generifs()

    pubmed_ids, genes = process_generifs()

    # process the abstracts file
    df = pd.read_csv('{}/generif.abstracts.txt'.format(DATA_DIRECTORY),
                     sep='\t',
                     header=None,
                     names=['pmid', 'abstract']
                     )
    df.drop_duplicates(inplace=True)

    # read the original file
    df_func = pd.read_csv('{}/generifs_basic.gz'.format(DATA_DIRECTORY),
                          sep='\t',
                          dtype={'PubMed ID (PMID) list': str}
                          )
    df_func = df_func.loc[df_func['#Tax ID'] == 9606]

    # merge the dataframes
    df = df.merge(df_func, how='inner', left_on='pmid', right_on='PubMed ID (PMID) list')

    # some abstracts could not be fetched
    df = df.loc[df['abstract'].notnull()]

    # TODO: placeholder, move this
    mg = get_client('gene')
    gene_info = mg.getgenes(set(genes), fields=["alias", "MIM", "symbol"], species="human")
    df = pd.DataFrame(gene_info)
    df.to_pickle('{}/gene_info.pkl'.format(DATA_DIRECTORY))

    # Get the Gene Symbol to remove them from the text
    gene_info = gene_info.set_index('query').to_dict(orient='index')

    df['gene_symbol'] = df['Gene ID'].apply(lambda geneid: gene_info[str(geneid)]['symbol'])
    df['gene_symbol'].fillna(0, inplace=True)
    df['gene_alias'] = df['Gene ID'].apply(lambda geneid: gene_info[str(geneid)]['alias'])
    df['gene_alias'].fillna(0, inplace=True)

    # remove gene symbols from abstracts (for the gene being referenced)
    df['abstract'] = df.apply(remove_gene_names, axis=1)

    # this is the input for training
    df[['pmid',
        'abstract',
        'Gene ID',
        'gene_symbol',
        'gene_alias']].to_csv('{}/generif.pubmed.abstracts.df.txt'.format(DATA_DIRECTORY),
                                                                            sep='\t'
                                                                            )
