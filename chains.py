from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from loader import load_and_prepare_data
from prompts import qa_prompt

load_dotenv()

# Load Data and Initialize Vector Store
db, embeddings = load_and_prepare_data()

# Initialize Retriever
retriever = db.as_retriever(
    search_type="mmr",  # Use Maximal Marginal Relevance
    search_kwargs={"k": 4, "fetch_k": 10}  # Retrieve top 3 diverse results from 10 candidates
)

# Initialize LLM
llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3-0324:free",
    base_url="https://openrouter.ai/api/v1"
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
