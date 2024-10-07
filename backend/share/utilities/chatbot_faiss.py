# /utilities/chatbot_faiss.py

import os
import time
import asyncio
import numpy as np
import faiss
import re

from langdetect import detect
from pythainlp.tokenize import sent_tokenize as thai_sent_tokenize
from nltk.tokenize import sent_tokenize as english_sent_tokenize

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
    """
        ChatbotFAISS processes user queries using FAISS for vector similarity search and caching.
        It handles both Thai and English languages by splitting inputs into individual questions
        and processing them asynchronously. 
        
        For future high-demand scenarios, it's recommended
        to move to a hybrid approach by integrating FAISS with scalable solutions like Redis
        or specialized vector databases to enhance performance and scalability.
    """
    def __init__(self):
        self.persist_directory = PERSIST_DIRECTORY
        self.pdf_path = PDF_PATH
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.MODEL_ID = MODEL_ID
        if not all([self.persist_directory, self.pdf_path, self.OPENAI_API_KEY, self.MODEL_ID]):
            log_controler.log_error("Required environment variables are not set.", "ChatbotFAISS __init__")
            raise ValueError("Required environment variables are not set.")
        try:
            # Initialize embeddings
            self.embeddings = self.initialize_embeddings()
            # Compute embedding dimension
            self.embedding_dimension = len(self.embeddings.embed_query("sample text"))
            # Initialize FAISS cache index
            self.cache_index = faiss.IndexFlatIP(self.embedding_dimension)
            self.cached_answers = []
            # Initialize vector store and QA chain
            self.vector_store = self.initialize_vector_store()
            self.qa_chain = self.initialize_qa_chain()
        except Exception as e:
            log_controler.log_error(f"Initialization failed: {e}", "ChatbotFAISS __init__")
            raise

    def split_into_sentences(self, text):
        # Split text into potential sentences using any delimiter (e.g., '?', '.', '!')
        potential_sentences = re.split(r'(?<=[.?!])\s*', text)
        sentences = []

        for sentence in potential_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                language = detect(sentence)
            except:
                language = 'en'
            if language == 'th':
                sentences.extend(thai_sent_tokenize(sentence))
            else:
                sentences.extend(english_sent_tokenize(sentence))
        return sentences


    def log_time(self, topic, description, start_time, end_time):
        """Logs the time used for a particular operation."""
        time_used = end_time - start_time
        log_message = f"{topic} | {description} | Time Used: {time_used:.2f} seconds"
        log_controler.log_info(log_message)

    def initialize_embeddings(self):
        topic = "Embeddings Initialization"
        description = "Initializing OpenAI embeddings"
        start_time = time.time()

        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)

        end_time = time.time()
        self.log_time(topic, description, start_time, end_time)
        return embeddings

    def load_and_split_pdf(self):
        topic = "PDF Processing"
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
        self.log_time(topic, description, start_time, end_time)
        return chunks

    def initialize_vector_store(self):
        topic = "FAISS Vector Store"
        description = "Creating or loading FAISS vector store"
        start_time = time.time()

        index_file = os.path.join(self.persist_directory, "index.faiss")

        try:
            if os.path.exists(index_file):
                log_controler.log_info("Loading existing FAISS vector store.")
                vector_store = FAISS.load_local(
                    self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
                log_controler.log_info("Loaded existing FAISS vector store.")
            else:
                log_controler.log_info("Creating new FAISS vector store.")
                document_chunks = self.load_and_split_pdf()
                vector_store = FAISS.from_documents(document_chunks, self.embeddings)
                vector_store.save_local(self.persist_directory)
                log_controler.log_info(f"Created new FAISS vector store | Persisted at: {self.persist_directory}")
            return vector_store
        except Exception as e:
            log_controler.log_error(f"Error initializing FAISS vector store: {e}", "initialize_vector_store")
            raise  # Re-raise to prevent proceeding with invalid vector_store
        finally:
            end_time = time.time()
            self.log_time(topic, description, start_time, end_time)

    def initialize_qa_chain(self):
        topic = "QA Chain Initialization"
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
            self.log_time(topic, description, start_time, end_time)
        return qa_chain

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def check_cache(self, question: str):
        try:
            if self.cache_index.ntotal == 0:
                return None, None
    
            embedding = self.embeddings.embed_query(question)
            embedding = np.array(embedding).astype('float32')
            # Normalize the embedding
            faiss.normalize_L2(embedding.reshape(1, -1))
            # Search in FAISS index
            D, I = self.cache_index.search(embedding.reshape(1, -1), k=1)
            if len(D[0]) > 0 and D[0][0] >= 0.8:
                cached_answer = self.cached_answers[I[0][0]]
                return cached_answer, "cache"
            return None, None
        except Exception as e:
            log_controler.log_error(f"Error checking cache: {e}", "check_cache")
            return None, None

    def add_to_cache(self, question: str, answer: str):
        try:
            embedding = self.embeddings.embed_query(question)
            embedding = np.array(embedding).astype('float32')
            # Normalize the embedding
            faiss.normalize_L2(embedding.reshape(1, -1))
            # Add to FAISS index
            self.cache_index.add(embedding.reshape(1, -1))
            # Store the answer
            self.cached_answers.append(answer)
            log_controler.log_info("Added question to FAISS cache index.")
        except Exception as e:
            log_controler.log_error(f"Error adding to cache: {e}", "add_to_cache")

    async def process_query(self, user_query: str) -> dict:
        topic = "User Query Processing"
        description = f"Processing user query"
        start_time = time.time()

        try:
            # Split the input into individual sentences/questions
            questions = self.split_into_sentences(user_query)
            tasks = []
            responses = []

            for question in questions:
                question = question.strip()
                if question:
                    # Process each question asynchronously
                    tasks.append(self.process_single_question(question))

            responses = await asyncio.gather(*tasks)

            # Combine the answers
            combined_answers = "\n".join([resp['answer'] for resp in responses if 'answer' in resp])

            # Determine overall type_res
            if all(resp.get('type_res') == 'cache' for resp in responses):
                overall_type_res = 'cache'
            else:
                overall_type_res = 'generate'

            response = {
                "answer": combined_answers,
                "type_res": overall_type_res
            }

            return response

        except Exception as e:
            log_controler.log_error(f"Error processing request: {str(e)}", "process_query")
            return {"error_code": "02", "msg": f"Error processing request: {str(e)}"}
        finally:
            end_time = time.time()
            self.log_time(topic, description, start_time, end_time)

    async def process_single_question(self, question: str) -> dict:
        # Check cache first
        cached_answer, cache_status = self.check_cache(question)
        if cached_answer:
            log_controler.log_info(f"Cache hit for query: {question}")
            return {
                "answer": cached_answer,
                "type_res": "cache"
            }

        # If not in cache, generate a new answer
        response_text = self.qa_chain.invoke(question)
        # Ensure response_text is a string
        if isinstance(response_text, dict):
            response_text = response_text.get("result", "")
            if not isinstance(response_text, str):
                log_controler.log_error(f"Unexpected response format: {response_text}", "process_single_question")
                return {"error_code": "03", "msg": "Unexpected response format from QA chain."}

        # Add the new Q&A to cache
        self.add_to_cache(question, response_text)

        return {
            "answer": response_text,
            "type_res": "generate"
        }