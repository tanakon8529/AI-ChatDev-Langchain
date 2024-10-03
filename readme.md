# AI ChatDev Langchain

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108.0-brightgreen.svg)
![Docker](https://img.shields.io/badge/Docker-20.10.7-blue.svg)

**AI ChatDev Langchain** is a robust and scalable chatbot application built using FastAPI, LangChain, and OpenAI's GPT models. It leverages FAISS for efficient vector storage and Redis for task queuing, providing a seamless conversational AI experience. The application is containerized using Docker and orchestrated with Docker Compose, ensuring easy deployment and scalability.

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
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Conversational AI**: Powered by OpenAI's GPT models for natural and context-aware responses.
- **Efficient Vector Storage**: Utilizes FAISS for fast similarity searches and vector storage.
- **Task Queuing**: Implements Redis for managing background tasks and queues.
- **Scalable Architecture**: Containerized with Docker and orchestrated using Docker Compose for easy scaling.
- **Structured Logging**: Comprehensive logging using a custom `LogControler` for monitoring and debugging.
- **API Security**: Implements OAuth2 for secure API access.

## Technologies Used

- **Programming Language**: Python 3.12
- **Web Framework**: FastAPI
- **AI & NLP**: LangChain, OpenAI GPT
- **Vector Database**: FAISS
- **Task Queue**: Redis
- **Containerization**: Docker, Docker Compose
- **Logging**: Custom LogControler
- **Testing**: Pytest

## Prerequisites

- **Docker**: Ensure Docker is installed on your machine. [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: Comes bundled with Docker Desktop. For Linux, follow [Docker Compose installation guide](https://docs.docker.com/compose/install/).
- **Git**: For cloning the repository. [Install Git](https://git-scm.com/downloads)

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

   # FastAPI AI Chat
   PORT_FASTAPI_AI_CHAT=8001
   API_PATH_FASTAPI_AI_CHAT=/api/chat

   # FastAPI OAuth2
   PORT_FASTAPI_OAUTH2=8000
   API_PATH_FASTAPI_OAUTH2=/api/oauth2

   # Redis
   PORT_REDIS=6379
   REDIS_PASSWORD=your_redis_password_here

   # Host Configuration
   HOST=localhost
   ```

   **Note**: Replace `your_redis_password_here` with a secure password.

3. **Verify `requirements.txt`**

   Ensure that `backend/share/requirements.txt` is updated and free from dependency conflicts. If you've followed the previous steps to resolve dependency issues, this file should be correctly configured.

## Configuration

- **Environment Variables**: All sensitive configurations and ports are managed via the `.env` file.
- **Docker Compose**: Defines services for FastAPI applications (`fastapi-oauth2` and `fastapi-ai-chat`) and Redis (`redis-queue`).

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

### 1. **FastAPI AI Chat**

- **Endpoint**: `/api/chat/query`
- **Method**: `POST`
- **Description**: Processes a user query and returns the chatbot's response.
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
      "response": "The total number of employees in 2022 is 1,234."
    }
    ```

  - **Error**:

    ```json
    {
      "error_code": "01",
      "msg": "OpenAI API Error: Detailed error message."
    }
    ```

### 2. **FastAPI OAuth2**

- **Endpoint**: `/api/oauth2/token`
- **Method**: `POST`
- **Description**: Obtains an OAuth2 token.
- **Request Body**:

  ```json
  {
    "username": "your_username",
    "password": "your_password"
  }
  ```

- **Response**:

  ```json
  {
    "access_token": "jwt_token_here",
    "token_type": "bearer"
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
│   └── tests/
├── docker-compose.yml
├── venv/
└── README.md
```

### **Key Directories and Files**

- **backend/**: Contains all backend services.
  - **fastapi-ai-chat/**: FastAPI service for AI Chat.
  - **fastapi-oauth2/**: FastAPI service handling OAuth2 authentication.
  - **share/**: Shared resources like configurations, utilities, and data.
    - **requirements.txt**: Python dependencies.
    - **utilities/**: Contains helper modules like `chatbot_faiss.py` and `log_controler.py`.
  - **tests/**: Contains test suites for the application.
- **docker-compose.yml**: Defines and orchestrates the Docker services.
- **venv/**: Python virtual environment (if used outside Docker).
- **README.md**: Project documentation.

## Testing

The project utilizes `pytest` for testing. To run the tests:

1. **Activate Virtual Environment**

   If you're testing outside of Docker, ensure your virtual environment is activated:

   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**

   Ensure all dependencies are installed:

   ```bash
   pip install -r backend/share/requirements.txt
   ```

3. **Run Tests**

   Navigate to the project root and execute:

   ```bash
   pytest
   ```

   **Sample Test Cases**:

   - **Health Check**

     ```python
     def test_health_check():
         response = client.get("/")
         assert response.status_code == 200
         assert response.json() == {"message": "Chatbot FAISS API is up and running."}
     ```

   - **Successful Query**

     ```python
     def test_query_success():
         payload = {"question": "What is the total number of employees in 2022?"}
         response = client.post("/query", json=payload)
         assert response.status_code == 200
         assert "response" in response.json()
         assert isinstance(response.json()["response"], str)
     ```

   - **Missing Question**

     ```python
     def test_query_missing_question():
         payload = {}
         response = client.post("/query", json=payload)
         assert response.status_code == 400
         assert response.json()["detail"] == "Question is required."
     ```

   - **Chatbot Not Initialized**

     ```python
     def test_query_chatbot_not_initialized(monkeypatch):
         # Simulate chatbot not initialized
         monkeypatch.setattr("apis.langgpt.mainmod.chat_bot", None)
         payload = {"question": "What is the total number of employees in 2022?"}
         response = client.post("/query", json=payload)
         assert response.status_code == 400
         assert response.json()["detail"] == "Chatbot is not initialized."
     ```

## Logging

The application uses a custom `LogControler` for structured and comprehensive logging. Logs are stored within the respective service directories and can be accessed via Docker volumes.

- **Log Files Location**:
  - **FastAPI OAuth2**: `backend/fastapi-oauth2/logs/`
  - **FastAPI AI Chat**: `backend/fastapi-ai-chat/logs/`

### **Log Levels**

- **INFO**: General operational messages that signify the application is working as expected.
- **ERROR**: Indicates a serious problem that has occurred, preventing part of the application from functioning.

### **Sample Log Entry**

```
Embeddings Initialization | Step 3/7: Initializing OpenAI embeddings | Time Used: 2.50 seconds
```

## Troubleshooting

### **1. Dependency Conflicts**

If you encounter dependency conflicts during Docker build, ensure that your `requirements.txt` specifies compatible versions. Refer to the [Dependencies Management](#dependencies-management) section.

### **2. FAISS Installation Issues**

FAISS can be challenging to install on certain architectures. Ensure you're using the correct FAISS version compatible with your system.

- **Alternative Installation via Conda**:

  ```bash
  conda install -c conda-forge faiss-cpu
  ```

### **3. Redis Authentication**

Ensure that the Redis password in your `.env` file matches the one specified in the `docker-compose.yml`.

### **4. Application Crashes on Startup**

Check the Docker container logs for detailed error messages:

```bash
docker logs -f <container_id>
```

Common issues might include missing environment variables, incorrect file paths, or unmet dependencies.

### **5. Docker Networking Issues**

Ensure that all services are connected to the correct Docker network (`web` in this case). Verify network configurations in `docker-compose.yml`.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

   Describe your changes and submit the pull request for review.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or support, please contact:

- **Email**: your.email@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)
