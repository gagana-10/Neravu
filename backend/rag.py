
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
        print("WARNING: Documents folder not found.")
        print(f"Expected location: {DOCUMENTS_DIR}")
        return documents

    for filename in sorted(os.listdir(DOCUMENTS_DIR)):

        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(DOCUMENTS_DIR, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read().strip()

            if content:
                documents.append({
                    "filename": filename,
                    "content": content
                })

        except Exception as e:
            print(f"WARNING: Could not read {filename}: {e}")

    return documents


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = load_documents()


# ============================================================
# BUILD TF-IDF RAG INDEX
# ============================================================

if documents:

    document_texts = [
        doc["content"]
        for doc in documents
    ]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2)
        )

        document_vectors = vectorizer.fit_transform(
            document_texts
        )

        print("=" * 60)
        print("RAG SYSTEM INITIALIZED")
        print(f"Documents loaded: {len(documents)}")
        print(f"Documents folder: {DOCUMENTS_DIR}")
        print("=" * 60)

    except Exception as e:

        print("ERROR: Could not initialize TF-IDF.")
        print(e)

        vectorizer = None
        document_vectors = None

else:

    vectorizer = None
    document_vectors = None

    print("=" * 60)
    print("WARNING: RAG SYSTEM HAS NO DOCUMENTS")
    print(f"Add .txt healthcare files to: {DOCUMENTS_DIR}")
    print("=" * 60)


# ============================================================
# SEARCH RELEVANT DOCUMENTS
# ============================================================

def retrieve_context(question, top_k=3):

    if not question or not question.strip():
        return "No healthcare question was provided."

    if not documents:
        return "No healthcare documents are available."

    if vectorizer is None or document_vectors is None:
        return "RAG system is not initialized."

    try:

        # Convert user question into TF-IDF vector
        question_vector = vectorizer.transform(
            [question]
        )

        # Calculate similarity between question
        # and every healthcare document
        similarities = cosine_similarity(
            question_vector,
            document_vectors
        )[0]

        # Rank documents from highest similarity
        # to lowest similarity
        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            score = float(similarities[index])

            # Ignore documents with zero relevance
            if score > 0:

                results.append({
                    "filename": documents[index]["filename"],
                    "content": documents[index]["content"],
                    "score": score
                })

        # ----------------------------------------------------
        # No relevant document found
        # ----------------------------------------------------

        if not results:
            return (
                "No relevant healthcare information was found "
                "in the available documents."
            )

        # ----------------------------------------------------
        # Build context for Gemini
        # ----------------------------------------------------

        context_parts = []

        for result in results:

            context_parts.append(
                f"SOURCE: {result['filename']}\n"
                f"RELEVANCE SCORE: {result['score']:.3f}\n"
                f"{result['content']}"
            )

        context = "\n\n".join(context_parts)

        return context

    except Exception as e:

        print(f"RAG retrieval error: {e}")

        return (
            "Unable to retrieve healthcare information "
            "from the documents."
        )


# ============================================================
# GET DOCUMENT INFORMATION
# ============================================================

def get_document_count():

    return len(documents)


def get_document_names():

    return [
        document["filename"]
        for document in documents
    ]


# ============================================================
# TEST RAG SYSTEM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("NERAVU HEALTHCARE RAG TEST")
    print("=" * 60)

    print(f"\nDocuments loaded: {len(documents)}")

    if documents:

        print("\nAvailable documents:")

        for document in documents:
            print(f"  - {document['filename']}")

    else:

        print("\nNo documents found.")
        print(f"Put your .txt files inside:")
        print(DOCUMENTS_DIR)

    question = input(
        "\nAsk a healthcare question: "
    ).strip()

    if question:

        print("\nSearching healthcare documents...\n")

        context = retrieve_context(question)

        print("=" * 60)
        print("RETRIEVED INFORMATION")
        print("=" * 60)

        print(context)

        print("\n" + "=" * 60)

    else:

        print("\nNo question entered.")

