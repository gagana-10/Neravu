import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOCUMENT LOCATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")


# ============================================================
# LOAD HEALTHCARE DOCUMENTS
# ============================================================

def load_documents():
    documents = []

    if not os.path.exists(DOCUMENTS_DIR):
        print("Documents folder not found.")
        return documents

    for filename in os.listdir(DOCUMENTS_DIR):

        if filename.endswith(".txt"):

            filepath = os.path.join(DOCUMENTS_DIR, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()

            documents.append({
                "filename": filename,
                "content": content
            })

    return documents


# ============================================================
# BUILD RAG RETRIEVER
# ============================================================

documents = load_documents()

if documents:
    document_texts = [doc["content"] for doc in documents]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    document_vectors = vectorizer.fit_transform(document_texts)

else:
    vectorizer = None
    document_vectors = None


# ============================================================
# SEARCH RELEVANT DOCUMENTS
# ============================================================

def retrieve_context(question, top_k=2):

    if not documents:
        return "No healthcare documents are available."

    if vectorizer is None:
        return "RAG system is not initialized."

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        if similarities[index] > 0:

            results.append({
                "filename": documents[index]["filename"],
                "content": documents[index]["content"],
                "score": float(similarities[index])
            })

    if not results:
        return "No relevant healthcare information was found."

    context = ""

    for result in results:

        context += (
            f"\nSOURCE: {result['filename']}\n"
            f"{result['content']}\n"
        )

    return context


# ============================================================
# TEST FUNCTION
# ============================================================

if __name__ == "__main__":

    print("Loading healthcare documents...")

    print(f"Documents loaded: {len(documents)}")

    question = input("\nAsk a healthcare question: ")

    print("\nSearching healthcare documents...\n")

    context = retrieve_context(question)

    print("========== RETRIEVED INFORMATION ==========")
    print(context)