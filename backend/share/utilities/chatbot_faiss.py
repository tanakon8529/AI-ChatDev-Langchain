# /utilities/chatbot_faiss.py

import os
import time
import asyncio
import numpy as np
import faiss

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from settings.configs import OPENAI_API_KEY, MODEL_ID, PERSIST_DIRECTORY, PDF_DIRECTORY_PATH, TEMPERATURE, BUILD_VECTOR_STORE

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
        self.pdf_directory_path = PDF_DIRECTORY_PATH  # Updated to handle directory
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.MODEL_ID = MODEL_ID
        self.TEMPERATURE = TEMPERATURE if TEMPERATURE else 0.3
        if not all([self.persist_directory, self.pdf_directory_path, self.OPENAI_API_KEY, self.MODEL_ID]):
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

        if BUILD_VECTOR_STORE == "True":
            self.clear_cache()
            self.rebuild_vector_store()

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

    def load_and_split_pdfs(self):
        topic = "PDF Processing"
        description = "Loading and splitting multiple PDFs into chunks"
        start_time = time.time()

        try:
            log_controler.log_info(f"Loading and splitting PDFs from directory: {self.pdf_directory_path}")
            full_directory_path = os.path.abspath(self.pdf_directory_path)
            log_controler.log_info(f"Full Directory Path: {full_directory_path}")

            if not os.path.isdir(full_directory_path):
                log_controler.log_error(f"PDF directory does not exist at path: {full_directory_path}", "load_and_split_pdfs")
                raise NotADirectoryError(f"PDF directory does not exist at path: {full_directory_path}")

            pdf_files = [file for file in os.listdir(full_directory_path) if file.lower().endswith('.pdf')]
            if not pdf_files:
                log_controler.log_error(f"No PDF files found in directory: {full_directory_path}", "load_and_split_pdfs")
                raise FileNotFoundError(f"No PDF files found in directory: {full_directory_path}")

            all_chunks = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(full_directory_path, pdf_file)
                log_controler.log_info(f"Loading PDF: {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                if not documents:
                    log_controler.log_error(f"No documents loaded from PDF: {pdf_path}", "load_and_split_pdfs")
                    continue  # Skip this PDF and continue with others

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                log_controler.log_info(f"Loaded and split PDF '{pdf_file}' into {len(chunks)} chunks.")

            if not all_chunks:
                log_controler.log_error("No chunks were created from any PDF files.", "load_and_split_pdfs")
                raise ValueError("No chunks were created from any PDF files.")

            log_controler.log_info(f"Total chunks created from all PDFs: {len(all_chunks)}")

        except Exception as e:
            log_controler.log_error(f"Error loading and splitting PDFs: {e}", "load_and_split_pdfs")
            raise  # Re-raise the exception to prevent proceeding with invalid chunks

        end_time = time.time()
        self.log_time(topic, description, start_time, end_time)
        return all_chunks

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
                document_chunks = self.load_and_split_pdfs()
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
        description = "Initializing RetrievalQA chain with AI Assistant role"
        start_time = time.time()

        try:
            prompt_template = """
                You are an AI Assistant for AP Thailand.
                Your knowledge is up to date until 2023.
                Your support languages are Thai and English.
                Use the following documents to answer the question.
                Only use the information from the documents to provide an accurate and concise answer.

                Documents:
                {context}

                Question:
                {question}

                Answer in the appropriate language (Thai or English), and ensure your response is accurate and helpful.
            """

            PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt_template
            )

            llm = ChatOpenAI(
                openai_api_key=self.OPENAI_API_KEY,
                model_name=self.MODEL_ID,
                temperature=self.TEMPERATURE
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            log_controler.log_info("Initialized RetrievalQA chain with AI Assistant role.")
        except Exception as e:
            log_controler.log_error(f"Error initializing QA chain: {e}", "initialize_qa_chain")
            raise  # Re-raise to prevent proceeding with invalid qa_chain
        finally:
            end_time = time.time()
            self.log_time(topic, description, start_time, end_time)
        return qa_chain

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
            similarity_score = D[0][0] if len(D[0]) > 0 else 0
            log_controler.log_info(f"Similarity score for cache check: {similarity_score}")
            if similarity_score >= 0.9:
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

    def clear_cache(self):
        self.cache_index = faiss.IndexFlatIP(self.embedding_dimension)
        self.cached_answers = []
        log_controler.log_info("Cleared FAISS cache index and cached answers.")

    def rebuild_vector_store(self):
        # Delete existing index file if exists
        index_file = os.path.join(self.persist_directory, "index.faiss")
        if os.path.exists(index_file):
            os.remove(index_file)
            log_controler.log_info(f"Deleted existing FAISS index file: {index_file}")
        
        # Re-initialize vector store
        self.vector_store = self.initialize_vector_store()
        log_controler.log_info("Rebuilt FAISS vector store.")

    def extract_questions(self, text):
        prompt_template = """
            You are an AI assistant that extracts individual questions from a user's input. The input may contain multiple questions in Thai or English, possibly in a single paragraph.

            Please list each question separately.

            Input:
            {text}

            Extracted Questions:
        """

        PROMPT = PromptTemplate(
            input_variables=["text"],
            template=prompt_template
        )

        llm = ChatOpenAI(
            openai_api_key=self.OPENAI_API_KEY,
            model_name=self.MODEL_ID,
            temperature=0  # Set temperature to 0 for deterministic output
        )

        chain = LLMChain(
            llm=llm,
            prompt=PROMPT
        )

        response = chain.run(text)
        # Split the response into individual questions
        questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return questions

    async def process_query(self, user_query: str) -> dict:
        topic = "User Query Processing"
        description = f"Processing user query"
        start_time = time.time()

        try:
            # Use AI to extract questions
            questions = self.extract_questions(user_query)
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
                "msg": "success",
                "data": {
                    "answer": combined_answers,
                    "type_res": overall_type_res
                }
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
        log_controler.log_info(f"Generating answer for query: {question}")
        response = self.qa_chain.invoke(question)

        # Log the raw response and source documents
        # log_controler.log_info(f"QA Chain response: {response['result']}")
        # log_controler.log_info(f"Source Documents: {response.get('source_documents', [])}")

        # Ensure response is a string
        if not isinstance(response['result'], str):
            log_controler.log_error(f"Unexpected response format: {response}", "process_single_question")
            return {"error_code": "03", "msg": "Unexpected response format from QA chain."}

        # Add the new Q&A to cache
        self.add_to_cache(question, response['result'])

        return {
            "answer": response['result'],
            "type_res": "generate"
        }

    def test_similarity_search(self, query: str):
        embedding = self.embeddings.embed_query(query)
        embedding = np.array(embedding).astype('float32')
        faiss.normalize_L2(embedding.reshape(1, -1))
        D, I = self.vector_store.index.search(embedding.reshape(1, -1), k=5)
        log_controler.log_info(f"Similarity Scores: {D}")
        log_controler.log_info(f"Indices of Top Matches: {I}")
