import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGPipeline:
    @staticmethod
    def _clean(v):
        """Strip whitespace and accidental quotes from secrets."""
        return v.strip().strip('"').strip("'").strip() if v else v

    def __init__(self):
        # Initialize Google gemini-embedding-2
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        self.collection_name = "rag_documents"
        
        qdrant_url = self._clean(os.environ.get("QDRANT_URL"))
        qdrant_api_key = self._clean(os.environ.get("QDRANT_API_KEY"))
        self.storage_mode = "memory"
        self.connection_error = None

        # Try Qdrant Cloud first; if it is unreachable (wrong/paused cluster, bad
        # URL, TLS problem) fall back to in-memory Qdrant instead of crashing.
        self.client = None
        if qdrant_url and qdrant_api_key:
            try:
                if "://" not in qdrant_url:
                    qdrant_url = "https://" + qdrant_url
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=20)
                client.get_collections()  # connectivity check
                self.client = client
                self.storage_mode = "cloud"
            except Exception as e:
                self.connection_error = f"{type(e).__name__}: {e}"
        if self.client is None:
            self.client = QdrantClient(location=":memory:")

        # Ensure collection exists in Qdrant
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
            
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        # Initialize Gemini 3.6 Flash LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

    def ingest_document(self, file_path):
        """Extracts text, chunks it, and upserts to Qdrant."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Splitting based on TRD specifications
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        # Upsert vectors along with payload metadata
        if splits:
            self.vector_store.add_documents(documents=splits)
        return len(splits)

    def answer_query(self, query):
        """Retrieves top K chunks and streams response from Gemini."""
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(query)
        
        def format_docs_with_metadata(docs):
            formatted_chunks = []
            for doc in docs:
                # PyPDFLoader stores page number in 'page' (0-indexed)
                page = doc.metadata.get('page', 0) + 1 
                formatted_chunks.append(f"[Page {page}]\n{doc.page_content}")
            return "\n\n".join(formatted_chunks)
            
        formatted_context = format_docs_with_metadata(retrieved_docs)
        
        prompt_template = """You are an expert AI assistant answering questions based on the provided context.
If the answer cannot be determined directly from the context, state "I cannot find this information in the uploaded document."

CONTEXT:
{retrieved_chunks}

USER QUESTION:
{user_query}"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["retrieved_chunks", "user_query"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Stream the response
        response_stream = chain.stream({
            "retrieved_chunks": formatted_context,
            "user_query": query
        })
        
        return response_stream, retrieved_docs
