from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# -------------------------
# Load dataset
# -------------------------
with open("college_data.txt", "r", encoding="utf-8") as f:
    data = f.read()

documents = data.split("\n\n")


# -------------------------
# Retriever
# -------------------------
class Retriever:
    def __init__(self, docs, top_k=2):
        self.docs = docs
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def retrieve(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(
            query_vector,
            self.doc_vectors
        ).flatten()

        top_indices = similarities.argsort()[::-1][:self.top_k]
        return [self.docs[i] for i in top_indices]


# -------------------------
# Generator using FLAN-T5
# -------------------------
class Generator:
    def __init__(self, max_tokens=80):
        self.pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            framework="pt"
        )
        self.max_tokens = max_tokens

    def generate(self, query, context):
        prompt = f"""
Answer the college-related question ONLY using the context below.
If answer is not available, say: Information not available.

Context:
{context}

Question:
{query}
"""

        output = self.pipe(
            prompt,
            max_new_tokens=self.max_tokens
        )

        return output[0]["generated_text"]


# -------------------------
# Chatbot
# -------------------------
class CollegeChatbot:
    def __init__(self, top_k=2, max_tokens=80):
        self.retriever = Retriever(documents, top_k)
        self.generator = Generator(max_tokens)

    def chat(self, query):
        retrieved_docs = self.retriever.retrieve(query)
        context = "\n".join(retrieved_docs)
        answer = self.generator.generate(query, context)
        return answer


# -------------------------
# Main program
# -------------------------
if __name__ == "__main__":
    top_k = int(input("Top K docs (1-5): "))
    max_tokens = int(input("Max tokens: "))

    bot = CollegeChatbot(top_k, max_tokens)

    print("\nCollege FAQ Chatbot")
    print("Type 'quit' to exit\n")

    while True:
        query = input("You: ")

        if query.lower() == "quit":
            break

        response = bot.chat(query)
        print("Bot:", response)