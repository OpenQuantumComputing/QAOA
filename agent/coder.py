# ----- Load API key ------
from dotenv import load_dotenv
load_dotenv()

import re
from pathlib import Path
from typing import Union, Optional
import io
from contextlib import redirect_stdout, redirect_stderr
import matplotlib

# ----- LangChain imports -----
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ----- Helper imports -----
from saveembedding import SaveEmbedding

repo_root = Path(__file__).resolve().parent.parent  # Go up to QAOA_Sanne root
DEFAULT_CONTEXT = [repo_root / "examples" / "MaxCut" / "KCutExamples.ipynb", repo_root / "qaoa" / "qaoa.py"]

class Coder:
    """
    A coding assistant that generates, executes, and improves Python code based on user queries.
    
    This class uses a conversational retrieval chain to maintain context and memory of previous interactions.
    It can generate code, analyze errors, and improve code based on feedback.
    It also captures media outputs from executed code for rendering in a chat interface.
    """
    def __init__(self, memory = None, context_files: Optional[list[Union[str, Path]]] = DEFAULT_CONTEXT, model:str="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0) # LLM.
        if memory is not None: # If a memory object is provided, use it (for sharing memory between instances).
            self.memory = memory
        else: # If no memory is provided, create a new memory object.
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",  # See prompt template for usage
                input_key="question",
                return_messages=True,
                max_token_limit=1000,  # Limit memory size to avoid excessive context
            )
        # Initialize vector store and retriever for context files (saved locally in embeddings to save tokens).
        embedding = SaveEmbedding(context_files, "Coder_embedding", "embeddings/Coder_embedding", "embeddings/Coder_cache")
        self.vectorstore = embedding.get_vectorstore()
        self.retriever = embedding.get_retriever()
        self.context = embedding.get_context()
        self._initialize_agent()

    def execute_code(self, code: str) -> str:
        """Executes the provided Python code and returns only error messages if any occur."""
        try:
            # Remove Markdown code fences if present
            code = re.sub(r"^```(?:python)?", "", code.strip(), flags=re.IGNORECASE)
            code = re.sub(r"```$", "", code.strip())

            # Redirect stdout to suppress circuit diagrams
            exec_globals = {}
            f = io.StringIO()

            with redirect_stdout(f), redirect_stderr(f):
                exec(code.strip(), exec_globals)

            # Only return success message if no errors
            return "SUCCESS: Code executed without errors"

        except Exception as e:
            # Return just the error type and message, not full traceback
            return f"ERROR: {type(e).__name__}: {str(e)}"

    def _initialize_agent(self) -> None:
        """Initialize and return the agent executor."""

        code_suggestion_prompt = PromptTemplate( # Prompt!!!
            input_variables=["context", "question"],
            template="""You are a AI, a Python coding assistant. 
            
You have four tasks based on the input:

1. If asked a query with no error information, generate Python code to solve the task.
2. If provided with an error message, analyze the error and suggest improvements to the code without generating new code.
3. If asked to improve code based on suggestions, generate improved Python code considering the provided feedback.
4. If an explaination is requested, provide a concise explanation.

Context:
{context}

Conversation history:
{chat_history}

Human: {question}
AI:

Guidelines:
1. Generate Python code to solve the task provided by the Human.
2. The code should be complete and executable
3. The code should include all necessary imports and mainly use the QAOA package. Don't include any imports that are not used.
4. The code should be formatted in markdown with ```python code fences
5. The code should include BRIEF comments in the code explaining key steps.
6. If analyzing an error, provide concise suggestions for improvement without generating code.
7. If analyzing an error and the error is 'NoneType', the function probably updates an internal variable, rather than returning a value.
In this case, try to find the variable that is updated and suggest using this instead.
8. Phrase all responses as if it is the first response to its corresponding query. i.e. don't mention executing code or changes made after executing code. 
9. After generating code, briefly explain the approach and any potential limitations in the code or discrepancies between the code and the task.
10. For parts of the task that are unspecified, provide brief reasoning for your choices.
11. Refer to previous tasks and responses in the conversation to maintain context and continuity.
""",
        )
        # Initialize chain that handles memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=(
                self.vectorstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )
                if self.vectorstore
                else None
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": code_suggestion_prompt,
            },
        )

    # Main function.
    def generate_and_test_code(self, query: str, max_iterations: int = 3) -> None:
        """Generate code, test it, and improve based on feedback."""
        current_code = None
        error_analysis = None

        for iteration in range(max_iterations): # Iterate up to max_iterations.
            print(f"\033[90m\n--- Iteration {iteration + 1} ---\033[0m")

            # Generate new or improve old code.
            if current_code is None:
                print("\033[90m\nGenerating initial response...\033[0m")
                result = self.qa_chain.invoke({"question": query})
            else:
                print("\033[90m\nImproving response based on previous error...\033[0m")
                result = self.qa_chain.invoke(
                    {
                        "question": f"Improve this code based on the following feedback: {error_analysis}\nOriginal task: {query}\nCode:\n{self._extract_code_block(current_code)}"
                    }
                )

            current_code = result["answer"]  # Extract response text.

            print("\033[90m\nResponse:\033[0m")
            print(f"\033[90m\n{current_code}\033[0m")

            # Execute the code if it contains a code block.
            if "```" in current_code:
                matplotlib.use(
                    "Agg"
                )  # Use a non-interactive backend for matplotlib (no verbose output in console).
                print("\033[96m\nExecuting code...\033[0m")
                execution_result = self.execute_code(
                    self._extract_code_block(current_code)
                )
                matplotlib.use("TkAgg")  # Reset to default backend
                print(f"\033[96m{execution_result}\033[0m")
                if "ERROR" in execution_result:
                    print(f"\033[96m\nGenerating error analysis...\033[0m")
                    result = self.qa_chain.invoke(
                        {
                            "question": f"""Analyze the following error: {execution_result}. 
                        Provide suggestions for improving the code without generating new code."""
                        }
                    )
                    error_analysis = result["answer"]
                    print(f"\033[96mError analysis: {error_analysis}\033[0m")
                else:
                    return current_code  # Return the response if no errors occurred.

            else: # If no code block is found, just return the response (nothing to test, we just trust it).
                return current_code

        print(f"\nReached maximum iterations ({max_iterations})")
        return current_code  # Return the last generated response when max tries are reached.

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown block."""
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()


if __name__ == "__main__":

    # Example usage.
    context_files = ["./examples/MaxCut/KCutExamples.ipynb", "./qaoa/qaoa.py"]
    assistant = Coder(context_files)

    # First query.
    query1 = "Create a qaoa instance using onehot encoding."
    print("\nFirst query: ")
    print(f"\033[1m{query1}\033[0m")
    final_code1 = assistant.generate_and_test_code(query1)
    print("\nFinal response to first query:")
    print(f"\033[1m{final_code1}\033[0m")

    # Second query relies on memory of the first one.
    # query2 = "Why did you choose the initial state and mixer like that?"
    # print("\nSecond query: ")
    # print(f"\033[1m{query2}\033[0m")
    # final_response2 = assistant.qa_chain.invoke({"question": query2})
    # print("\nFinal response to second query:")
    # print(f"\033[1m{final_response2["answer"]}\033[0m")

    # # Third query.
    # query3 = """Create a qaoa circuit solving the max k-cut problem with k = 3 for this 10-node graph using binary encoding and the full hamiltonian:
    #     graph [
    # node [
    #     id 0
    #     label "0"
    # ]
    # node [
    #     id 1
    #     label "1"
    # ]
    # node [
    #     id 2
    #     label "2"
    # ]
    # node [
    #     id 3
    #     label "3"
    # ]
    # node [
    #     id 4
    #     label "4"
    # ]
    # node [
    #     id 5
    #     label "5"
    # ]
    # node [
    #     id 6
    #     label "6"
    # ]
    # node [
    #     id 7
    #     label "7"
    # ]
    # node [
    #     id 8
    #     label "8"
    # ]
    # node [
    #     id 9
    #     label "9"
    # ]
    # edge [
    #     source 0
    #     target 4
    #     weight 0.3246074330296992
    # ]
    # edge [
    #     source 0
    #     target 7
    #     weight 0.6719596645247027
    # ]
    # edge [
    #     source 1
    #     target 4
    #     weight 0.5033779645445525
    # ]
    # edge [
    #     source 1
    #     target 5
    #     weight 0.8197417437657258
    # ]
    # edge [
    #     source 1
    #     target 6
    #     weight 0.1689752608979167
    # ]
    # edge [
    #     source 2
    #     target 4
    #     weight 0.8578794331926194
    # ]
    # edge [
    #     source 2
    #     target 5
    #     weight 0.10889087475274517
    # ]
    # edge [
    #     source 2
    #     target 6
    #     weight 0.29609241287667165
    # ]
    # edge [
    #     source 2
    #     target 7
    #     weight 0.3385595778596342
    # ]
    # edge [
    #     source 2
    #     target 9
    #     weight 0.49871018015134483
    # ]
    # edge [
    #     source 3
    #     target 4
    #     weight 0.5646214337732219
    # ]
    # edge [
    #     source 3
    #     target 5
    #     weight 0.22675259631935551
    # ]
    # edge [
    #     source 3
    #     target 6
    #     weight 0.42653644275637315
    # ]
    # edge [
    #     source 3
    #     target 8
    #     weight 0.9458888986056379
    # ]
    # edge [
    #     source 4
    #     target 5
    #     weight 0.6274516118216547
    # ]
    # edge [
    #     source 4
    #     target 7
    #     weight 0.6461631361850252
    # ]
    # edge [
    #     source 4
    #     target 8
    #     weight 0.07077280704236999
    # ]
    # edge [
    #     source 4
    #     target 9
    #     weight 0.061962597519110374
    # ]
    # edge [
    #     source 5
    #     target 6
    #     weight 0.12115603714424517
    # ]
    # edge [
    #     source 5
    #     target 7
    #     weight 0.6596288196387271
    # ]
    # edge [
    #     source 5
    #     target 8
    #     weight 0.8184188538157214
    # ]
    # edge [
    #     source 5
    #     target 9
    #     weight 0.686461546892179
    # ]
    # edge [
    #     source 7
    #     target 8
    #     weight 0.5014423218237379
    # ]
    # edge [
    #     source 8
    #     target 9
    #     weight 0.11336603624375363
    # ]
    # ]
    # Visualize both the graph and the circuit."""
    # print("\nThird query: ")
    # print(f"\033[1m{query3}\033[0m")
    # final_code3 = assistant.generate_and_test_code(query3)
    # print("\nFinal response to third query:")
    # print(f"\033[1m{final_code3}\033[0m")

    # # Print memory summary
    # print("\033[95m\nMemory summary:\033[0m")
    # print(f"\033[95m{assistant.memory.buffer}\033[0m")