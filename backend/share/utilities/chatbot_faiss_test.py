# /utilities/chatbot_faiss_test.py

import random
from typing import List, Dict

from utilities.chatbot_faiss import ChatbotFAISS
from utilities.log_controler import LogControler
from utilities.question_generator import QuestionGenerator  # Utility to generate questions


class ChatbotFAISSTest:
    """
    ChatbotFAISSTest is designed to test the ChatbotFAISS class by:
    1. Generating a specified number of questions.
    2. Analyzing the accuracy of the chatbot's answers against provided answers.
    """

    def __init__(self):
        self.chatbot = ChatbotFAISS()
        self.log_controler = LogControler()
        self.question_generator = QuestionGenerator()

    def generate_questions(self, number_of_questions: int, topic: str = "general") -> List[str]:
        """
        Generates a specified number of questions based on the given topic.

        Args:
            number_of_questions (int): The number of questions to generate.
            topic (str): The topic or category for question generation.

        Returns:
            List[str]: A list of generated questions.
        """
        try:
            self.log_controler.log_info(
                f"Generating {number_of_questions} questions on topic: '{topic}'"
            )
            questions = self.question_generator.generate(number_of_questions, topic)
            self.log_controler.log_info(
                f"Generated {len(questions)} questions successfully."
            )
            return questions
        except Exception as e:
            self.log_controler.log_error(
                f"Error generating questions: {e}",
                "ChatbotFAISSTest.generate_questions"
            )
            return []

    async def analyze_accuracy(self, questions_and_answers: Dict[str, str]) -> Dict[str, float]:
        """
        Analyzes the accuracy of the chatbot's answers against the provided answers.

        Args:
            questions_and_answers (Dict[str, str]): A dictionary where keys are questions and values are the expected answers.

        Returns:
            Dict[str, float]: A dictionary containing each question and its corresponding accuracy percentage.
        """
        accuracy_results = {}
        try:
            self.log_controler.log_info("Starting accuracy analysis...")
            for question, expected_answer in questions_and_answers.items():
                self.log_controler.log_info(f"Processing question: {question}")
                response = await self.chatbot.process_query(question)
                if 'data' in response and 'answer' in response['data']:
                    chatbot_answer = response['data']['answer']
                    accuracy = self.calculate_similarity(expected_answer, chatbot_answer)
                    accuracy_results[question] = accuracy
                    self.log_controler.log_info(
                        f"Question: {question} | Accuracy: {accuracy:.2f}%"
                    )
                else:
                    accuracy_results[question] = 0.0
                    self.log_controler.log_error(
                        f"Failed to get a valid answer for question: {question}",
                        "ChatbotFAISSTest.analyze_accuracy"
                    )
            self.log_controler.log_info("Completed accuracy analysis.")
        except Exception as e:
            self.log_controler.log_error(
                f"Error analyzing accuracy: {e}",
                "ChatbotFAISSTest.analyze_accuracy"
            )
        return accuracy_results

    def calculate_similarity(self, expected: str, actual: str) -> float:
        """
        Calculates the similarity percentage between the expected and actual answers.

        Args:
            expected (str): The expected answer.
            actual (str): The chatbot's actual answer.

        Returns:
            float: The similarity percentage (0.0 to 100.0).
        """
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        return ratio * 100.0

    async def run_tests(self, number_of_questions: int, topic: str = "general") -> None:
        """
        Runs the test by generating questions and analyzing their accuracy.

        Args:
            number_of_questions (int): The number of questions to generate and test.
            topic (str): The topic or category for question generation.
        """
        # Step 1: Generate Questions
        questions = self.generate_questions(number_of_questions, topic)

        if not questions:
            self.log_controler.log_error("No questions generated. Aborting tests.", "ChatbotFAISSTest.run_tests")
            return

        # Step 2: Retrieve Expected Answers from FAISS Vector Store
        questions_and_expected_answers = {}
        for q in questions:
            expected_answer = self.retrieve_expected_answer(q)
            questions_and_expected_answers[q] = expected_answer

        # Step 3: Analyze Accuracy
        accuracy_results = await self.analyze_accuracy(questions_and_expected_answers)

        # Step 4: Summarize Results
        self.summarize_results(accuracy_results)

    def retrieve_expected_answer(self, question: str) -> str:
        """
        Retrieves the expected answer for a given question from the FAISS vector store.

        Args:
            question (str): The question for which to retrieve the expected answer.

        Returns:
            str: The expected answer retrieved from the FAISS vector store.
        """
        try:
            self.log_controler.log_info(
                f"Retrieving expected answer for question: {question}"
            )
            # Perform similarity search with k=1
            results = self.chatbot.vector_store.similarity_search(question, k=1)
            if results:
                # Assume the first result is the most relevant
                expected_answer = results[0].page_content.strip()
                self.log_controler.log_info(
                    f"Retrieved expected answer for question: {question}"
                )
                return expected_answer
            else:
                self.log_controler.log_info(
                    f"No relevant documents found for question: {question}"
                )
                return "No relevant answer found."
        except Exception as e:
            self.log_controler.log_error(
                f"Error retrieving expected answer for question '{question}': {e}",
                "ChatbotFAISSTest.retrieve_expected_answer"
            )
            return "Error retrieving expected answer."

    async def process_single_question(self, question: str) -> dict:
        try:
            # Check cache first
            cached_answers, cache_status = self.chatbot.check_cache(question)
            if cached_answers:
                self.log_controler.log_info(f"Cache hit for question: {question}")
                # Select a random answer from the cached list
                answer = random.choice(cached_answers)
                return {
                    "answer": answer,
                    "type_res": "cache"
                }

            self.log_controler.log_info(
                f"Cache miss for question: {question}. Generating new answer."
            )
            # If not in cache, generate a new answer
            response = self.chatbot.qa_chain.invoke(question)

            # Ensure response is a string
            if not isinstance(response['result'], str):
                self.log_controler.log_error(
                    f"Unexpected response format: {response}",
                    "ChatbotFAISSTest.process_single_question"
                )
                return {"error_code": "03", "msg": "Unexpected response format from QA chain."}

            answer = response['result'].strip()

            # Check if the answer is meaningful
            if not answer or any(phrase in answer.lower() for phrase in [
                "i'm sorry", "sorry", "ขออภัย", "ไม่มีข้อมูล", "ไม่พบข้อมูล",
                "i could not find", "no information", "no data", "no results",
                "not found", "ไม่มีคำตอบ", "ไม่มีข้อมูลที่ต้องการ", "ไม่พบข้อมูลที่ต้องการ",
                "no specific mention", "no specific information", "no specific data", "no specific results",
                "I cannot find", "I cannot locate", "I cannot provide", "I cannot answer",
                "I cannot retrieve", "I can't find", "I can't locate", "I can't provide",
                "I can't answer", "I can't retrieve",
            ]):
                return {
                    "answer": answer,
                    "type_res": "no_answer"
                }

            # Add the new Q&A to cache
            self.log_controler.log_info(
                f"Adding new answer to cache for question: {question}"
            )
            self.chatbot.add_to_cache(question, answer)

            return {
                "answer": answer,
                "type_res": "generate"
            }
        except Exception as e:
            self.log_controler.log_error(
                f"Error processing single question: {e}",
                "ChatbotFAISSTest.process_single_question"
            )
            return {"error_code": "04", "msg": f"Error processing question: {str(e)}"}

    def summarize_results(self, accuracy_results: Dict[str, float]) -> None:
        """
        Summarizes and logs the accuracy results.

        Args:
            accuracy_results (Dict[str, float]): A dictionary containing each question and its accuracy percentage.
        """
        total_questions = len(accuracy_results)
        if total_questions == 0:
            self.log_controler.log_info("No accuracy results to summarize.")
            return

        total_accuracy = sum(accuracy_results.values())
        average_accuracy = total_accuracy / total_questions

        high_accuracy = [q for q, acc in accuracy_results.items() if acc >= 80.0]
        low_accuracy = [q for q, acc in accuracy_results.items() if acc < 80.0]

        self.log_controler.log_info(f"Total Questions Tested: {total_questions}")
        self.log_controler.log_info(f"Average Accuracy: {average_accuracy:.2f}%")
        self.log_controler.log_info(
            f"Questions with High Accuracy (>=80%): {len(high_accuracy)}"
        )
        self.log_controler.log_info(
            f"Questions with Low Accuracy (<80%): {len(low_accuracy)}"
        )

        # Optionally, log detailed results
        for question, accuracy in accuracy_results.items():
            self.log_controler.log_info(
                f"Question: {question} | Accuracy: {accuracy:.2f}%"
            )
