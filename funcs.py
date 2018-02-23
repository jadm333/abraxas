from gensim.models import Phrases

def quitarpuntuacion(df):
    df = df.copy(deep=True)
    df['comment_text'] = df['comment_text'].str.replace('"', '')
    ##Qitar espaciondel principio y el final
    df['comment_text'] = df['comment_text'] .str.strip()
    ## Quitar Puntuacion
    df['comment_text'] = df['comment_text'].str.replace('@', '')
    df['comment_text'] = df['comment_text'].str.replace('#', '')
    df['comment_text'] = df['comment_text'].str.replace('!', '')
    df['comment_text'] = df['comment_text'].str.replace('¿', '')
    df['comment_text'] = df['comment_text'].str.replace('?', '')
    df['comment_text'] = df['comment_text'].str.replace("'", '')
    df['comment_text'] = df['comment_text'].str.replace('"', '')
    df['comment_text'] = df['comment_text'].str.replace('%', '')
    df['comment_text'] = df['comment_text'].str.replace('+', '')
    df['comment_text'] = df['comment_text'].str.replace('=', '')
    df['comment_text'] = df['comment_text'].str.replace('`', '')
    df['comment_text'] = df['comment_text'].str.replace('~', '')
    df['comment_text'] = df['comment_text'].str.replace('|', '')
    df['comment_text'] = df['comment_text'].str.replace(';', '')
    df['comment_text'] = df['comment_text'].str.replace('.', '')
    df['comment_text'] = df['comment_text'].str.replace('á', '')
    df['comment_text'] = df['comment_text'].str.replace('à', '')
    df['comment_text'] = df['comment_text'].str.replace('é', '')
    df['comment_text'] = df['comment_text'].str.replace('.', '')
    df['comment_text'] = df['comment_text'].str.replace('‘', '')
    df['comment_text'] = df['comment_text'].str.replace('“', '')
    df['comment_text'] = df['comment_text'].str.replace('”', '')
    df['comment_text'] = df['comment_text'].str.replace('¡', '')
    df['comment_text'] = df['comment_text'].str.replace('º', '')
    df['comment_text'] = df['comment_text'].str.replace('ª', '')
    df['comment_text'] = df['comment_text'].str.replace(',', ' ')
    df['comment_text'] = df['comment_text'].str.replace('.', ' ')
    df['comment_text'] = df['comment_text'].str.replace('/', '')
    df['comment_text'] = df['comment_text'].str.replace(':', ' ')
    df['comment_text'] = df['comment_text'].str.replace(':', ' ')
    df['comment_text'] = df['comment_text'].str.replace('*', ' ')
    df['comment_text'] = df['comment_text'].str.replace('(', ' ')
    df['comment_text'] = df['comment_text'].str.replace(')', ' ')
    df['comment_text'] = df['comment_text'].str.replace('{', ' ')
    df['comment_text'] = df['comment_text'].str.replace('}', ' ')
    df['comment_text'] = df['comment_text'].str.replace(']', ' ')
    df['comment_text'] = df['comment_text'].str.replace('[', ' ')
    df['comment_text'] = df['comment_text'].str.strip()
    df['comment_text'] = df['comment_text'].str.lower()
    return df

def entidades(docs,nlp):
    processed_docs = []
    for doc in nlp.pipe(docs, n_threads=8, batch_size=10000):
        ents = doc.ents  

        doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

        doc.extend([str(entity) for entity in ents if len(entity) > 1])
        
        processed_docs.append(doc)

    return processed_docs

def bigramas(docs):
    bigram = Phrases(docs, min_count=20)
    doc = docs
    for idx in range(len(doc)):
        for token in bigram[doc[idx]]:
            if '_' in token:
                doc[idx].append(token)
    return doc

