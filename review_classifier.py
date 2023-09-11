from categories import category_summaries
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings


def load_file(file_path):
    """
    loads the file 
    returns chunks of documents as list
    """
    
    loader = TextLoader(file_path)
    pages = loader.load_and_split()
    return pages


def get_embeddings(input_pages):
    """
    takes list of documents
    returns chroma vectorstore
    """
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(input_pages, embeddings)

    return db
    

def get_similar_category(txt_emb, all_categories):
    """
    takes text embeddings and list of all the summaries
    returns the most matched summary (dictionary) based on similarity score
    """
    scores = {}
    for category, summary in all_categories.items():
        matching_docs_with_score = txt_emb.similarity_search_with_score(summary)
        scores[category] = matching_docs_with_score[0][-1]

    min_score = min(scores, key=scores.get)

    most_similar = {
        min_score: all_categories[min_score]
    }
    print(most_similar)

    return most_similar


if __name__ == "__main__":
    file_docs = load_file("reviews_files/allergy.txt")
    text_embeddings = get_embeddings(file_docs)
    similar_summary = get_similar_category(text_embeddings, category_summaries)