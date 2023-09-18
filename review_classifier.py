from categories import category_summaries
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings


def load_and_embed(file_path):
    """
    loads the file 
    returns db of documents and embeddings
    """
    
    loader = TextLoader(file_path)
    pages = loader.load_and_split()
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(pages, embeddings)

    return db
    

def get_similar_category(txt_emb, all_categories):
    """
    takes text embeddings and list of all the summaries
    returns the most matched summary (dictionary) based on similarity score
    """
    scores = []
    result_dict = {}
    for category, summary in all_categories.items():

        matching_score = txt_emb.similarity_search_with_score(summary, k=1)
        score = matching_score[0][-1]
        category_to_score = {
            category: score
        }
        scores.append(category_to_score)
    
    score = sorted(scores, key=lambda x: list(x.values()))[0]

    score_key = list(score.keys())[0]
    result_dict[score_key] = all_categories[score_key]
    
    print(result_dict)

    return result_dict


if __name__ == "__main__":

    directory_path = "review_files"

    file_embeddings = load_and_embed(directory_path + "/" + "food_quality_1.txt")
    similar_summary = get_similar_category(file_embeddings, category_summaries)

