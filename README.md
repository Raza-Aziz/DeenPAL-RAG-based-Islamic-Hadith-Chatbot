# DeenPAL-RAG-based-Islamic-Hadith-Chatbot

![GitHub repo size](https://img.shields.io/github/repo-size/Raza-Aziz/DeenPAL-RAG-based-Islamic-Hadith-Chatbot)
![GitHub contributors](https://img.shields.io/github/contributors/Raza-Aziz/DeenPAL-RAG-based-Islamic-Hadith-Chatbot)
![GitHub stars](https://img.shields.io/github/stars/Raza-Aziz/DeenPAL-RAG-based-Islamic-Hadith-Chatbot?style=social)

> A Retrieval-Augmented Generation (RAG) based chatbot designed to provide accurate and personalized medical information to users.

## 🚀 Features

- **Personalized Responses:** Delivers tailored information based on user queries.
- **Reliable Sources:** Utilizes trusted hadith sources to ensure information accuracy.
- **Interactive Interface:** Engages users through a conversational platform for seamless interaction.

## 🛠️ Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/Raza-Aziz/DeenPAL-RAG-based-Islamic-Hadith-Chatbot.git
   cd DeenPAL-RAG-based-Islamic-Hadith-Chatbot
   ```

2. **Set up a virtual environment:**

   ```sh
   conda create -n deen-pal python=3.10 -y
   conda activate deen-pal   # Alternative to above line
   ```    
   
3. **Create a `.env` file in the root directory and add your DEEPSEEK or any LLM API key as follows:**

    ```bash
    DEEPSEEK_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

4. **Install dependencies:**
    All the dependencies are in colab_requirements.txt, so you can install them all by simply running

   ```bash
   pip install -r colab_requirements.txt
   ```

   I use uv package manager, cause it works a bit faster.
   ```bash
    pip install uv
    uv pip install -r colab_requirements.txt
   ```

## 📌 Usage

1. **Prepare the data:**

   - Place your Hadith documents in the `data/` directory in PDF format. I used Sahih Muslim and Sahih Bukhari books (all volumes of both) as my data.

2. **Run the chatbot:**

   ```bash
   streamlit run app.py
   ```

3. **Access the chatbot interface:**

   Open your web browser and navigate to `http://localhost:8080`.

## 🧐 Explanation

The **DeenPAL-RAG-based-Islamic-Hadith-Chatbot** follows a **Retrieval-Augmented Generation (RAG)** approach to provide responses based on Islamic Hadiths. Here's a structured explanation of its key components:

### 1\. **Data Preparation (`loader.py`)**

-   **Loading Hadith PDFs:** The chatbot loads Hadith PDFs stored in the `data/` directory.

-   **Metadata Processing:** Extracts and structures metadata from the Hadith documents.

-   **Text Splitting:**

    ```python
    pattern = r"(?:Chapter\s\d+:)|(?:Book\s\d+,\sNumber\s\d+:)"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[pattern], is_separator_regex=True
    )
    chunks = text_splitter.split_documents(documents)
    ```

    This splits the Hadith text into chunks based on hadith, meaning each hadith will be a chunk. This helped in retaining semantic instead of a fixed-size character text splitting.

-   **Downloading Embedding Model & Initializing Vector Store:**

    ```python
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory='database/chroma_db'
    )
    ```

    The embeddings are generated and stored in **ChromaDB** for efficient retrieval.

-   **Caching Data Loading with `@st.cache_resource`:**

    ```py
    @st.cache_resource
    def load_and_prepare_data():
    ```

    This ensures that data is **only loaded once** per app session, reducing unnecessary processing. Otherwise, the whole data loading, embedding, intializing vector store processes will be executed with each user query.

### 2\. **Retrieval & Generation Pipeline (`chains.py`)**

-   **Loading Data & Initializing Vector Store:** Retrieves pre-processed Hadiths and embeddings.

-   **Retriever Setup:**

    ```python
    retriever = db.as_retriever(
        search_type="mmr",  # Use Maximal Marginal Relevance
        search_kwargs={"k": 4, "fetch_k": 10}  # Retrieve top 4 diverse results from 10 candidates
    )
    ```
    -   **Maximal Marginal Relevance (MMR)** is used **instead of similarity-based retrieval**.

    -  I tried both, and `MMR` **increased diversity** in retrieved Hadiths while maintaining relevance.
    Whereas using `similarity_score_threshold` was making the Chatbot give the same hadiths redundantly.   

    -  It helps the **LLM provide different but relevant Hadiths** instead of similar ones, offering a broader context and a more precise answer to user queries.

-   **Language Model Initialization:**

    ```py
    llm = ChatOpenAI(
        model="deepseek/deepseek-chat-v3-0324:free",
        base_url="https://openrouter.ai/api/v1"
    )
    ```

    The chatbot utilizes an OpenAI-compatible model via OpenRouter API to generate responses.

-   **Combining Retrieval & Generation:** Uses LangChain to form a retrieval-based response pipeline.


### 3\. **Prompt Design (`prompts.py`)**

-   **QA System Prompt:**

    ```py
    qa_system_prompt = (
        "You are an Islamic religious assistant for accurately retrieving hadiths..."
        "{context}"
    )
    ```

    This instructs the AI to retrieve and **present Hadiths with citations**, followed by a **brief explanation** before answering the question concisely.

-   **Chat Prompt Template:**

    ```py
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    ```

    This ensures structured communication between the system and the user.

### 4\. **User Interface (`app.py`)**

-   **Streamlit-based Web UI:** Provides a chat interface for user interaction.
-   **Maintains Chat History:** Uses `st.session_state` to store past messages.
-   **Handles User Queries:** When a user submits a question, it:
    1.  Displays the question.
    2.  Processes the input through the RAG pipeline.
    3.  Generates and displays an AI response.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Contact

For any questions, feel free to reach out:

- GitHub: [@Raza-Aziz](https://github.com/Raza-Aziz)
- Email: razaaziz9191@gmail.com


