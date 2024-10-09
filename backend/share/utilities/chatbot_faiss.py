# /utilities/chatbot_faiss.py

import os
import time
import asyncio
import numpy as np
import faiss
import re

from typing import List
from langdetect import detect
from pythainlp.tokenize import sent_tokenize as thai_sent_tokenize
from nltk.tokenize import sent_tokenize as english_sent_tokenize

from difflib import SequenceMatcher
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from settings.configs import OPENAI_API_KEY, MODEL_ID, PERSIST_DIRECTORY, \
                        PDF_DIRECTORY_PATH, TEMPERATURE, BUILD_VECTOR_STORE, \
                        CLEAR_CACHE

from utilities.log_controler import LogControler

# Initialize the LogControler
log_controler = LogControler()

class NormalizedOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_query(self, text: str) -> List[float]:
        embedding = super().embed_query(text)
        embedding = np.array(embedding).astype('float32')
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding.flatten().tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = super().embed_documents(texts)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        return embeddings.tolist()

class ChatbotFAISS:
    """
        ChatbotFAISS processes user queries using FAISS for vector similarity search and caching.
        It handles both Thai and English languages by splitting inputs into individual questions
        and processing them asynchronously.
    """
    def __init__(self):
        self.persist_directory = PERSIST_DIRECTORY
        self.pdf_directory_path = PDF_DIRECTORY_PATH
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
            # Initialize vector store and QA chain
            self.vector_store = self.initialize_vector_store()
            self.qa_chain = self.initialize_qa_chain()
            # Initialize cache with normalized embeddings
            index = faiss.IndexFlatIP(self.embedding_dimension)
            self.cache_docstore = InMemoryDocstore({})
            self.cache_vector_store = FAISS(
                index=index,
                docstore=self.cache_docstore,
                index_to_docstore_id={},
                embedding_function=self.embeddings
            )
        except Exception as e:
            log_controler.log_error(f"Initialization failed: {e}", "ChatbotFAISS __init__")
            raise

        if BUILD_VECTOR_STORE == "True":
            self.clear_cache()
            self.rebuild_vector_store()
        if CLEAR_CACHE == "True":
            self.clear_cache()

    def log_time(self, topic, description, start_time, end_time):
        """Logs the time used for a particular operation."""
        time_used = end_time - start_time
        log_message = f"{topic} | {description} | Time Used: {time_used:.2f} seconds"
        log_controler.log_info(log_message)

    def initialize_embeddings(self):
        topic = "Embeddings Initialization"
        description = "Initializing OpenAI embeddings"
        start_time = time.time()

        embeddings = NormalizedOpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)

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
    
    def are_questions_similar(self, question1: str, question2: str) -> bool:
        ratio = SequenceMatcher(None, question1.strip(), question2.strip()).ratio()
        return ratio > 0.95  # Adjust the threshold as needed

    def check_cache(self, question: str):
        try:
            if not self.cache_vector_store.docstore._dict:
                return None, None

            # Perform similarity search in cache vector store
            docs_and_scores = self.cache_vector_store.similarity_search_with_score(question, k=1)
            if docs_and_scores:
                doc, similarity = docs_and_scores[0]  # 'similarity' is the inner product (cosine similarity)
                log_controler.log_info(f"Similarity score for cache check: {similarity}")
                cached_question = doc.metadata.get('question', '')
                cached_answer = doc.page_content
                if similarity >= 0.9:
                    # Compare the questions directly
                    if self.are_questions_similar(question, cached_question):
                        return cached_answer, "cache"
            return None, None
        except Exception as e:
            log_controler.log_error(f"Error checking cache: {e}", "check_cache")
            return None, None

    def add_to_cache(self, question: str, answer: str):
        try:
            # Create a document with the question and answer as metadata
            doc = Document(
                page_content=answer,
                metadata={'question': question}
            )
            # Add the document to the FAISS cache vector store
            self.cache_vector_store.add_documents([doc])
            log_controler.log_info("Added question and answer to FAISS cache index.")
        except Exception as e:
            log_controler.log_error(f"Error adding to cache: {e}", "add_to_cache")

    def clear_cache(self):
        # Reinitialize the FAISS index for caching
        index = faiss.IndexFlatIP(self.embedding_dimension)
        self.cache_docstore = InMemoryDocstore({})
        self.cache_vector_store = FAISS(
            index=index,
            docstore=self.cache_docstore,
            index_to_docstore_id={},
            embedding_function=self.embeddings
        )
        log_controler.log_info("Cleared FAISS cache vector store.")

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
        """
            Extracts questions from the input text.
            Thai and English questions are supported.
        """
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

    async def process_query(self, user_query: str) -> dict:
        topic = "User Query Processing"
        description = "Processing user query"
        count = 1
        total_steps = 3
        step_count = f"Step {count}/{total_steps}"
        start_time = time.time()

        try:
            # Step 1: Extract Questions
            log_controler.log_info(f"{topic} | {step_count} | Extracting questions.")
            self.log_time(topic, description, start_time, time.time())

            questions = self.extract_questions(user_query)
            if not questions:
                log_controler.log_info(f"{topic} | {step_count} | No questions found in the input.")
                return {
                    "msg": "No questions found in the input.",
                    "data": {
                        "answer": "",
                        "type_res": "no_questions"
                    }
                }

            # Step 2: Process Each Question Asynchronously
            count += 1
            step_count = f"Step {count}/{total_steps}"
            log_controler.log_info(f"{topic} | {step_count} | Processing questions.")

            tasks = []
            for idx, question in enumerate(questions, start=1):
                question = question.strip()
                if question:
                    log_controler.log_info(f"{topic} | {step_count} | Processing question {idx}/{len(questions)}: {question}")
                    tasks.append(self.process_single_question(question))

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Step 3: Combine Answers
            count += 1
            step_count = f"Step {count}/{total_steps}"
            combined_answers = ""
            type_res_list = []
            for resp in responses:
                if isinstance(resp, Exception):
                    log_controler.log_error(f"{topic} | {step_count} | Error processing question: {resp}", "process_query")
                    combined_answers += "An error occurred while processing one of the questions.\n"
                else:
                    combined_answers += resp.get('answer', '') + "\n"
                    type_res_list.append(resp.get('type_res'))

            # Determine Overall type_res
            overall_type_res = 'cache' if all(tr == 'cache' for tr in type_res_list) else 'generate'

            response = {
                "msg": "success",
                "data": {
                    "answer": combined_answers.strip(),
                    "type_res": overall_type_res
                }
            }

            log_controler.log_info(f"{topic} | {step_count} | Combined Answers: {combined_answers.strip()}")
            return response

        except Exception as e:
            log_controler.log_error(f"{topic} | {step_count} | Error processing request: {str(e)}", "process_query")
            return {"error_code": "02", "msg": f"Error processing request: {str(e)}"}

        finally:
            # Ensure that the final step is logged
            if count < total_steps:
                count = total_steps
                step_count = f"Step {count}/{total_steps}"
            self.log_time(f"{topic} | {step_count}", description, start_time, time.time())

    async def process_single_question(self, question: str) -> dict:
        # Check cache first
        cached_answer, cache_status = self.check_cache(question)
        if cached_answer:
            return {
                "answer": cached_answer,
                "type_res": "cache"
            }

        # If not in cache, generate a new answer
        response = self.qa_chain.invoke(question)

        # Ensure response is a string
        if not isinstance(response['result'], str):
            log_controler.log_error(f"Unexpected response format: {response}", "process_single_question")
            return {"error_code": "03", "msg": "Unexpected response format from QA chain."}

        answer = response['result'].strip()

        # Check if the answer is meaningful
        if not answer or any(phrase in answer.lower() for phrase in [
            "i'm sorry", "sorry", "ขออภัย", "ไม่มีข้อมูล", "ไม่พบข้อมูล", "i could not find", "no information",
            "no data", "no results", "not found", "ไม่มีคำตอบ", "ไม่มีข้อมูลที่ต้องการ", "ไม่พบข้อมูลที่ต้องการ"
            "no specific mention", "no specific information", "no specific data", "no specific results",
            "I cannot find", "I cannot locate", "I cannot provide", "I cannot answer", "I cannot retrieve",
            "I cannot", "I can't find", "I can't locate", "I can't provide", "I can't answer", "I can't retrieve",
        ]):
            log_controler.log_info(f"No valid answer generated for query: {question}")
            log_controler.log_info(f"No valid answer generated: {answer}")
            return {
                "answer": answer,
                "type_res": "no_answer"
            }

        # Add the new Q&A to cache
        self.add_to_cache(question, answer)

        return {
            "answer": answer,
            "type_res": "generate"
        }

    def test_similarity_search(self, query: str):
        embedding = self.embeddings.embed_query(query)
        embedding = np.array(embedding).astype('float32')
        faiss.normalize_L2(embedding.reshape(1, -1))
        D, I = self.vector_store.index.search(embedding.reshape(1, -1), k=5)
        log_controler.log_info(f"Similarity Scores: {D}")
        log_controler.log_info(f"Indices of Top Matches: {I}")
