import os
import weave
import asyncio
import json
import warnings
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from weave import Model

warnings.filterwarnings("ignore")

weave.init(project_name="genai-application")
persist_directory = "Vector_db"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

@weave.op()
def generate_solution_with_llm(query: str, vector_db_content: str, model: str ):
    user_input = f"User query: {query}. Retrieved relevant data: {vector_db_content}"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Based on the query and content, provide the solution.",
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content

class RAGModel(Model):
    model_name: str = "llama3-8b-8192"

    @weave.op()
    def predict(self, question: str) -> dict:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectorstore_retriever = vectordb.as_retriever(search_kwargs={"k": 1})
        retrieved_documents = vectorstore_retriever.get_relevant_documents(question)
        retrieved_content = " ".join([doc.page_content for doc in retrieved_documents])
        answer = generate_solution_with_llm(
            query=question,
            vector_db_content=retrieved_content,
            model=self.model_name,
        )
        return {'answer': answer, 'context': retrieved_content}

# Define your scoring function
@weave.op()
async def context_precision_score(question, model_output):
    context_precision_prompt = """Given the question, answer, and context, verify if the context was useful in arriving at the given answer. Give a verdict as "1" if useful and "0" if not, with JSON output.
Output in only valid JSON format.

question: {question}
context: {context}
answer: {answer}
verdict: """

    prompt = context_precision_prompt.format(
        question=question,
        context=model_output['context'],
        answer=model_output['answer'],
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model="llama-3.1-70b-versatile",
    )
    
    response_message = response.choices[0].message.content.strip()

    try:
        response_json = json.loads(response_message)
        verdict = int(response_json["verdict"]) == 1
    except (json.JSONDecodeError, KeyError, ValueError):
        verdict = False

    return {"verdict": verdict}

questions = [
    {"question": "How does MEMORAG improve memory integration in RAG models?"},
    {"question": "What challenges arise when using memory in knowledge discovery for MEMORAG?"},
    {"question": "How can MEMORAG enhance context retention in complex tasks?"},
    {"question": "How does MEMORAG handle outdated knowledge in dynamic datasets?"},
    {"question": "What is the role of memory management in scaling MEMORAG systems?"}
]

model = RAGModel()

evaluation = weave.Evaluation(dataset=questions, scorers=[context_precision_score])

asyncio.run(evaluation.evaluate(model))
