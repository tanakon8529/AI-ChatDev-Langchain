# /utilities/chatbot_faiss.py

import os
import time
import openai

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from settings.configs import OPENAI_API_KEY, MODEL_ID, PERSIST_DIRECTORY, PDF_PATH

from utilities.log_controler import LogControler

# Initialize the LogControler
log_controler = LogControler()

class ChatbotFAISS:
    def __init__(self):
        self.persist_directory = PERSIST_DIRECTORY
        self.pdf_path = PDF_PATH
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.MODEL_ID = MODEL_ID
        # Check if the required environment variables are set
        if not self.persist_directory or not self.pdf_path or not self.OPENAI_API_KEY or not self.MODEL_ID:
            log_controler.log_error("Required environment variables are not set.", "ChatbotFAISS __init__")
            raise ValueError("Required environment variables are not set.")

        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.total_steps = 7

        try:
            # Initialize embeddings
            self.embeddings = self.initialize_embeddings(self.OPENAI_API_KEY)
            # Initialize vector store
            self.vector_store = self.initialize_vector_store()
            # Initialize QA chain
            self.qa_chain = self.initialize_qa_chain()
        except Exception as e:
            log_controler.log_error(f"Initialization failed: {e}", "ChatbotFAISS __init__")
            raise  # Re-raise the exception to prevent running with incomplete initialization

    def log_step(self, topic, step_num, description, start_time, end_time):
        """Logs the step information using LogControler."""
        time_used = end_time - start_time
        log_message = f"{topic} | Step {step_num}/{self.total_steps}: {description} | Time Used: {time_used:.2f} seconds"
        log_controler.log_info(log_message)

    def initialize_embeddings(self, OPENAI_API_KEY):
        topic = "Embeddings Initialization"
        step_num = 3
        description = "Initializing OpenAI embeddings"
        start_time = time.time()

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        end_time = time.time()
        self.log_step(topic, step_num, description, start_time, end_time)
        return embeddings

    def load_and_split_pdf(self):
        topic = "PDF Processing"
        step_num = 4
        description = "Loading and splitting PDF into chunks"
        start_time = time.time()

        try:
            log_controler.log_info(f"Loading and splitting PDF: {self.pdf_path}")
            full_pdf_path = os.path.abspath(self.pdf_path)
            log_controler.log_info(f"Full PDF Path: {full_pdf_path}")

            if not os.path.isfile(full_pdf_path):
                log_controler.log_error(f"PDF file does not exist at path: {full_pdf_path}", "load_and_split_pdf")
                raise FileNotFoundError(f"PDF file does not exist at path: {full_pdf_path}")

            loader = PyPDFLoader(full_pdf_path)
            documents = loader.load()

            if not documents:
                log_controler.log_error("No documents loaded from PDF.", "load_and_split_pdf")
                raise ValueError("No documents loaded from PDF.")

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            log_controler.log_info(f"Loaded and split PDF into {len(chunks)} chunks.")

        except Exception as e:
            log_controler.log_error(f"Error loading and splitting PDF: {e}", "load_and_split_pdf")
            raise  # Re-raise the exception to prevent proceeding with invalid chunks

        end_time = time.time()
        self.log_step(topic, step_num, description, start_time, end_time)
        return chunks

    def initialize_vector_store(self):
        topic = "FAISS Vector Store"
        step_num = 5
        description = "Creating or loading FAISS vector store"
        start_time = time.time()

        try:
            if os.path.exists(self.persist_directory):
                log_controler.log_info("Loading existing FAISS vector store.")
                vector_store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
                log_controler.log_info("Loaded existing FAISS vector store.")
            else:
                log_controler.log_info("Creating new FAISS vector store.")
                document_chunks = self.load_and_split_pdf()
                vector_store = FAISS.from_documents(document_chunks, self.embeddings)
                vector_store.save_local(self.persist_directory)
                log_controler.log_info("Created new FAISS vector store.")
            return vector_store
        except Exception as e:
            log_controler.log_error(f"Error initializing FAISS vector store: {e}", "initialize_vector_store")
            raise  # Re-raise to prevent proceeding with invalid vector_store
        finally:
            end_time = time.time()
            self.log_step(topic, step_num, description, start_time, end_time)

    def initialize_qa_chain(self):
        topic = "QA Chain Initialization"
        step_num = 6
        description = "Initializing RetrievalQA chain"
        start_time = time.time()

        try:
            llm = ChatOpenAI(
                openai_api_key=self.OPENAI_API_KEY,
                model_name=self.MODEL_ID,
                temperature=0.3
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=False
            )
            log_controler.log_info("Initialized RetrievalQA chain.")
        except Exception as e:
            log_controler.log_error(f"Error initializing QA chain: {e}", "initialize_qa_chain")
            raise  # Re-raise to prevent proceeding with invalid qa_chain
        finally:
            end_time = time.time()
            self.log_step(topic, step_num, description, start_time, end_time)
        return qa_chain

    def process_query(self, user_query: str) -> str:
        topic = "User Query Processing"
        step_num = 7
        description = f"Processing user query: {user_query}"
        start_time = time.time()

        try:
            response = self.qa_chain.invoke(user_query)
            log_controler.log_info(f"User Query: {user_query} | Response: {response}")
            return response
        except openai.APIResponseValidationError as e:
            log_controler.log_error(f"OpenAI API Error: {e}", "process_query")
            return {"error_code": "01", "msg": f"OpenAI API Error: {e}"}
        except Exception as e:
            log_controler.log_error(f"Error processing request: {str(e)}", "process_query")
            return {"error_code": "02", "msg": f"Error processing request: {str(e)}"}
        finally:
            end_time = time.time()
            time_used = end_time - start_time
            log_message = f"{topic} | Step {step_num}/{self.total_steps}: {description} | Time Used: {time_used:.2f} seconds"
            log_controler.log_info(log_message)
