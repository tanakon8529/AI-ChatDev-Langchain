
# AI ChatDev Langchain

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108.0-brightgreen.svg)
![Docker](https://img.shields.io/badge/Docker-20.10.7-blue.svg)

**AI ChatDev Langchain** is a robust and scalable chatbot application built using FastAPI, LangChain, and OpenAI's GPT models. It leverages FAISS for efficient vector storage and Redis for task queuing, providing a seamless conversational AI experience. The application is containerized using Docker and orchestrated with Docker Compose, ensuring easy deployment and scalability. Additionally, it includes comprehensive testing utilities to ensure the chatbot's accuracy and reliability.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Logging](#logging)
- [License](#license)

## Features

- **Conversational AI**: Powered by OpenAI's GPT models for natural and context-aware responses.
- **Efficient Vector Storage**: Utilizes FAISS for fast similarity searches and vector storage.
- **Task Queuing**: Implements Redis for managing background tasks and queues.
- **Scalable Architecture**: Containerized with Docker and orchestrated using Docker Compose for easy scaling.
- **Structured Logging**: Comprehensive logging using a custom `LogControler` for monitoring and debugging.
- **API Security**: Implements OAuth2 for secure API access.
- **Automated Testing**: Includes `ChatbotFAISSTest` for generating questions and analyzing chatbot accuracy against the FAISS vector store.

## Technologies Used

- **Programming Language**: Python 3.12
- **Web Framework**: FastAPI
- **AI & NLP**: LangChain, OpenAI GPT
- **Vector Database**: FAISS
- **Task Queue**: Redis
- **Containerization**: Docker, Docker Compose
- **Logging**: Custom `LogControler`
- **Testing**: `chatbot_faiss_test.py` utilizing asynchronous processing and FAISS vector store

## Prerequisites

- **Docker**: Ensure Docker is installed on your machine. [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: Comes bundled with Docker Desktop. For Linux, follow [Docker Compose installation guide](https://docs.docker.com/compose/install/).
- **Git**: For cloning the repository. [Install Git](https://git-scm.com/downloads)
- **Python 3.12**: If you plan to run components outside Docker. [Download Python](https://www.python.org/downloads/)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ai-chatdev-langchain.git
   cd ai-chatdev-langchain
   ```

2. **Set Up Environment Variables**

   Create a `.env` file in the root directory and populate it with the necessary environment variables. Here's a sample template:

   ```dotenv
   # .env

   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key
   MODEL_ID=gpt-4

   # FAISS Configuration
   PERSIST_DIRECTORY=path_to_persist_directory
   PDF_DIRECTORY_PATH=path_to_pdf_directory

   # Chatbot Configuration
   TEMPERATURE=0.7
   BUILD_VECTOR_STORE=True
   CLEAR_CACHE=False

   # Redis Configuration
   REDIS_HOST=redis
   REDIS_PORT=6379
   REDIS_PASSWORD=your_redis_password_here

   # FastAPI Configuration
   PORT_FASTAPI_AI_CHAT=8001
   API_PATH_FASTAPI_AI_CHAT=/api/chat
   PORT_FASTAPI_OAUTH2=8000
   API_PATH_FASTAPI_OAUTH2=/api/oauth2

   # Host Configuration
   HOST=localhost
   ```

   **Note**: Replace placeholders like `your_openai_api_key`, `path_to_persist_directory`, and `your_redis_password_here` with your actual configuration values.

3. **Verify `requirements.txt`**

   Ensure that `backend/share/requirements.txt` is updated and includes all necessary dependencies. If you've followed the previous steps to resolve dependency issues, this file should be correctly configured.

## Configuration

- **Environment Variables**: All sensitive configurations and ports are managed via the `.env` file.
- **Docker Compose**: Defines services for FastAPI applications (`fastapi-oauth2` and `fastapi-ai-chat`), Redis (`redis-queue`), and other utilities.
- **FAISS Vector Store**: Configured to persist vectors for efficient similarity searches.

## Running the Application

1. **Build and Start Services**

   From the root directory, run:

   ```bash
   docker-compose up --build
   ```

   This command will build the Docker images and start all services defined in the `docker-compose.yml` file.

2. **Access the APIs**

   - **FastAPI OAuth2**: Accessible at `http://localhost:8000/api/oauth2`
   - **FastAPI AI Chat**: Accessible at `http://localhost:8001/api/chat`
   - **Redis**: Accessible at `localhost:6379` with the password specified in `.env`

3. **Health Check**

   To verify that the services are running, navigate to:

   - **FastAPI OAuth2 Health Check**: `http://localhost:8000/`
   - **FastAPI AI Chat Health Check**: `http://localhost:8001/`

   You should receive a JSON response confirming that the API is up and running.

## API Endpoints

### 1. **Ask AI LangChain GPT**

- **Endpoint**: `/v1/ask/`
- **Method**: `POST`
- **Description**: Processes a user query and returns the chatbot's response.
- **Authentication**: Requires a valid access token via OAuth2.
- **Request Body**:

  ```json
  {
    "question": "What is the total number of employees in 2022?"
  }
  ```

- **Response**:

  - **Success**:

    ```json
    {
        "msg": "success",
        "data": {
            "answer": "In 2022, AP Thailand and its subsidiaries had a total of 2,808 employees, including 18 executives and 58 temporary staff.",
            "type_res": "generate"
        }
    }
    ```

  - **Error**:

    ```json
    {
      "error_code": "01",
      "msg": "OpenAI API Error: Detailed error message."
    }
    ```

### 2. **Test AI LangChain GPT**

- **Endpoint**: `/v1/test/`
- **Method**: `GET`
- **Description**: Initiates the testing process by generating questions and analyzing the chatbot's accuracy.
- **Authentication**: Requires a valid access token via OAuth2.
- **Response**:

  - **Success**:

    ```json
    {
        "msg": "Testing initiated successfully.",
        "data": {
            "total_questions": 10,
            "average_accuracy": 85.50,
            "high_accuracy_questions": 8,
            "low_accuracy_questions": 2
        }
    }
    ```

  - **Error**:

    ```json
    {
      "error_code": "02",
      "msg": "Error processing request: Detailed error message."
    }
    ```

## Project Structure

```
AI-ChatDev-Langchain/
├── LICENSE
├── backend/
│   ├── fastapi-ai-chat/
│   │   ├── Dockerfile
│   │   ├── logs/
│   │   ├── data/
│   │   └── ... (other files and directories)
│   ├── fastapi-oauth2/
│   │   ├── Dockerfile
│   │   ├── logs/
│   │   └── ... (other files and directories)
│   ├── share/
│   │   ├── core/
│   │   ├── data/
│   │   ├── middlewares/
│   │   ├── requirements.txt
│   │   ├── settings/
│   │   └── utilities/
│   │       ├── bot_profiles.py
│   │       ├── chatbot_faiss.py
│   │       ├── chatbot_faiss_test.py
│   │       ├── cookie_controler.py
│   │       ├── hash_controler.py
│   │       ├── json_controler.py
│   │       ├── log_controler.py
│   │       ├── nltk_handler.py
│   │       ├── question_generator.py
│   │       ├── redis_connector.py
│   │       ├── time_controler.py
│   │       ├── characters_controler.py
│   │       ├── directory_controler.py
│   │       ├── cookie_controler.py
│   │       └── ... (other utility modules)
│   └── tests/
│       └── ... (test suites and scripts)
├── docker-compose.yml
├── readme.md
└── venv/
```

### **Key Directories and Files**

- **backend/**: Contains all backend services.
  - **fastapi-ai-chat/**: FastAPI service for AI Chat with endpoints `/v1/ask/` and `/v1/test/`.
  - **fastapi-oauth2/**: FastAPI service handling OAuth2 authentication and token management.
  - **share/**: Shared resources like configurations, utilities, and data.
    - **requirements.txt**: Python dependencies for the backend services.
    - **utilities/**: Contains helper modules such as:
      - **`chatbot_faiss.py`**: Core chatbot functionality using FAISS for vector storage.
      - **`chatbot_faiss_test.py`**: Testing utilities for generating questions and analyzing chatbot accuracy.
      - **`question_generator.py`**: Generates questions using LangChain.
      - **`log_controler.py`**: Custom logging controller for structured logging.
      - **Other utility modules**: Manage various functionalities like profiling, hashing, JSON handling, etc.
  - **tests/**: Contains test suites and scripts for automated testing.

- **docker-compose.yml**: Defines and orchestrates the Docker services, including FastAPI applications and Redis.
- **venv/**: Python virtual environment (if used outside Docker).
- **README.md**: Project documentation.

## Testing

### **ChatbotFAISSTest**

The `ChatbotFAISSTest` class is designed to evaluate the accuracy and reliability of the `ChatbotFAISS` class. It performs the following operations:

1. **Generating Questions**: Utilizes `QuestionGenerator` to create a specified number of insightful questions based on a given topic.
2. **Retrieving Expected Answers**: Fetches expected answers from the FAISS vector store to ensure consistency with the knowledge base.
3. **Analyzing Accuracy**: Compares the chatbot's responses to the expected answers using similarity metrics.
4. **Summarizing Results**: Logs a comprehensive summary of the accuracy analysis, including average accuracy and categorization of questions based on their accuracy scores.

### **Running Tests**

1. **Navigate to the Backend Directory**

   ```bash
   cd backend/share/utilities/
   ```

2. **Activate Virtual Environment (If Not Using Docker)**

   ```bash
   source ../../venv/bin/activate  # Adjust the path as necessary
   ```

3. **Install Testing Dependencies**

   Ensure that all necessary packages for testing are installed. You might need to install additional packages like `sentence-transformers` if you've enhanced similarity measures.

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the Test Script**

   Create a test script or use an existing one to instantiate and run `ChatbotFAISSTest`. Here's an example script:

   ```python
   # /backend/share/utilities/run_tests.py

   import asyncio
   from chatbot_faiss_test import ChatbotFAISSTest

   async def main():
       tester = ChatbotFAISSTest()
       number_of_questions = 10  # Specify the number of questions
       topic = "Artificial Intelligence"  # Specify the topic
       await tester.run_tests(number_of_questions, topic)

   if __name__ == "__main__":
       asyncio.run(main())
   ```

   **Run the Test Script:**

   ```bash
   python run_tests.py
   ```

   This will generate questions, retrieve expected answers from the FAISS vector store, analyze the chatbot's responses, and log the accuracy results.

## Logging

The application uses a custom `LogControler` for structured and comprehensive logging. Logs are stored within the respective service directories and can be accessed via Docker volumes or directly from the file system.

- **Log Files Location**:
  - **FastAPI OAuth2**: `backend/fastapi-oauth2/logs/`
  - **FastAPI AI Chat**: `backend/fastapi-ai-chat/logs/`
  - **Utilities**: Logs related to utilities like `chatbot_faiss_test.py` are typically stored in their respective directories.

### **Log Levels**

- **INFO**: General operational messages that signify the application is working as expected.
- **ERROR**: Indicates a serious problem that has occurred, preventing part of the application from functioning.

### **Sample Log Entry**

```
Embeddings Initialization | Initializing OpenAI embeddings | Time Used: 2.50 seconds
```

## License

This project is licensed under the [MIT License](LICENSE).
