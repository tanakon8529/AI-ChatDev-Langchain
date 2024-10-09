# /utilities/question_generator.py

from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from utilities.log_controler import LogControler
from settings.configs import OPENAI_API_KEY, MODEL_ID

class QuestionGenerator:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model_id = MODEL_ID
        self.log_controler = LogControler()
        
        # Initialize LangChain's ChatOpenAI
        try:
            self.llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model_name=self.model_id,
                temperature=0.7  # Adjust temperature as needed
            )
            self.log_controler.log_info("Initialized ChatOpenAI successfully.")
        except Exception as e:
            self.log_controler.log_error(f"Failed to initialize ChatOpenAI: {e}", "QuestionGenerator.__init__")
            raise

        # Define a prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["number_of_questions", "topic"],
            template=(
                "Generate {number_of_questions} insightful and diverse questions on the topic of {topic}. "
                "Ensure that the questions are clear, concise, and cover various subtopics within the main topic."
            )
        )

    def generate(self, number_of_questions: int, topic: str = "general") -> List[str]:
        """
        Generates a specified number of questions based on the given topic using LangChain's ChatOpenAI.

        Args:
            number_of_questions (int): The number of questions to generate.
            topic (str): The topic or category for question generation.

        Returns:
            List[str]: A list of generated questions.
        """
        try:
            self.log_controler.log_info(f"Generating {number_of_questions} questions on topic: '{topic}'")
            
            # Format the prompt with the given parameters
            prompt = self.prompt_template.format(number_of_questions=number_of_questions, topic=topic)
            
            # Create a HumanMessage (or SystemMessage if you want to set system instructions)
            messages = [
                HumanMessage(content=prompt)
            ]
            
            # Call the LLM to generate the questions
            response = self.llm(messages)
            
            # Extract and process the response
            questions_text = response.content.strip()
            # Split the questions assuming they are separated by newlines or numbered
            questions = self._parse_questions(questions_text, number_of_questions)
            
            self.log_controler.log_info(f"Generated {len(questions)} questions successfully.")
            return questions
        except Exception as e:
            self.log_controler.log_error(f"Error generating questions with LangChain: {e}", "QuestionGenerator.generate")
            return []

    def _parse_questions(self, text: str, expected_count: int) -> List[str]:
        """
        Parses the generated text into individual questions.

        Args:
            text (str): The raw text output from the LLM.
            expected_count (int): The expected number of questions.

        Returns:
            List[str]: A list of cleaned questions.
        """
        import re

        # Attempt to split by lines
        lines = text.split('\n')
        questions = []
        for line in lines:
            # Remove numbering if present (e.g., "1. What is...")
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if cleaned.endswith('?'):
                questions.append(cleaned)
            if len(questions) >= expected_count:
                break
        
        # If not enough questions, attempt splitting by other delimiters
        if len(questions) < expected_count:
            # Split by question marks and re-add them
            split_questions = re.split(r'\?+', text)
            questions = [q.strip() + '?' for q in split_questions if q.strip()]
            questions = questions[:expected_count]
        
        return questions
