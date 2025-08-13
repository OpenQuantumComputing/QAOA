# ----- Imports -----
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import init_chat_model

# ----- Helper imports -----
from explainer import Explainer
from coder import Coder

class Planner:
    """
    A planning agent that generates plans for QAOA code or explanations based on user input. It then sends the plan to either the Explainer or Coder agent for further processing.

    Attributes:
        model (str): The language model to use for planning. Default is "gpt-4.1".
        temperature (float): The temperature for the language model. Default is 0.

    Methods:
        __init__(model, temperature): Initializes the planner with an LLM and a planning prompt.
        plan(description): Generates a plan based on the user description and stored context.
        __call__(description): Calls the planner with a description.
        set_context(): Sets or updates the context variable with relevant documentation.
    """

    def __init__(self, model="openai:gpt-4.1", temperature=0):
        """Initialize the planning agent with an LLM, a planning prompt, and optional context.

        Args:
            model (str): The language model to use for planning. Default is "openai:gpt-4.1".
            temperature (float): The temperature for the language model. Default is 0.
        """
        # Initialize the language model
        self.llm = init_chat_model(model, temperature=temperature)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
            max_token_limit=1000,
        )
        self.context = ""
        self.set_context()
        self.prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""
You are an expert in the QAOA Python package and a code assistant to the USER. Your task is to create a plan (a prompt) for another AGENT to follow.

You have access to the following context: {context}, which contains the valid options for the QAOA package. The context is divided into three parts:
    1. Valid Initial states
    2. Valid Mixers
    3. Valid Problems

You have also access to the chat history: {chat_history}

Given the USER's input: "{question}":

***** RULES *****
ONLY do 1 of the following 3 cases:
--- CASE 0: Validate USER input with the context.
    - If any USER-specified initial state, problem, or mixer is NOT an exact match with an option in {context}. Then:
     - Respond ONLY with: "The [initial state/problem/mixer] '[name]' is not a valid option based on the documentation." and the list of valid options for the QAOA package provided by the context.
     - Do NOT generate anything else, no code, no plans, no titles, no other explanations.
     - You will not use a template for this case, but rather a direct response.
    - Else, continue to the next case.

Example of CASE 0:
 - USER input: "could you make a qaoa example with the sanne problem, the x mixer and the plus initial state?"
 - Context: {context}
 - Response: "The problem 'sanne' is not a valid option based on the documentation."

--- CASE 1: Generate a plan to create code using the QAOA package.
1. If the USER requests specific code or how to implement QAOA, generate a numbered list of concise, implementation-focused steps to create a Python script using only the QAOA package and only valid intial states/mixers/problems which are explicitly written in the context. 
 - The title above the steps are ALWAYS "CASE 1: Plan to generate code using the QAOA package".
 - Do NOT write any code or call any tools.  
 - Only describe how to do each step.

    The step template you will use if CASE 1 is selected:
        1. Answer this query: {question}
 
 - ALWAYS include the steps 1 if CASE 1 applies.
 - IF the USER asks for a visualization, include a step that asks for this. If the USER asks for something specific to the visualization, include a step that asks for this.
 - IF the USER asks for a cost landscape, include a step that asks for this. 

--- CASE 2: Generate a plan to explain the QAOA package.
2. If the USER asks for an explanation about the QAOA package, generate a concise TWO-WORD list of components. It should be written as bullet points.
 - Title of the list is either "CASE 2: Plan over which components of the QAOA package to explain" or "CASE 2: Plan for explanation of structures and relationships in the QAOA package".
 - Use this case if: "explain", "explanation", "components", "parts", or "structure" is in the description and "code" or "implementation" is not.
 - Include only the components that are relevant to the USER's request.
   
   Some the components that can be relevant:
         - QAOA class
         - Problems classes
         - Mixers classes
         - Initial states classes

    If the USER asks for a "structure", "relation", "relationship", "overview" or "hierarchy" of the QAOA package then it can be relevant to include:
         - structure
         - relation
         - overview

 - IF the USER ONLY asks for a specific class, method, or variable, then ONLY include that class, method, or variable in the list you generate. It does not need to be in the context to be included.

Strictly follow these rules:  
 - Output ONLY the numbered list of steps or the list of components.  
 - No additional commentary, no code, no explanations, no tool calls.  
 - If you cannot follow these rules, respond with: "Cannot provide a plan based on the input."

Remember: Be concise, focused, and precise.
""",
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

        # Initialize the other agents Explainer and Coder with context files
        self.other_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
            max_token_limit=1000,
        )
        
        # Initialize the Explainer and Coder with the same memory
        self.explainer = Explainer(self.other_memory, embedding=True)
        self.coder = Coder(self.other_memory)

    def plan(self, description: str) -> str:
        """Generate a plan based on the user description and stored context."""
        result = self.chain.invoke(
            {"question": description, "context": self.context}
        )
        # Extract the plan from the result
        plan = result["text"]
        low_plan = plan.lower()

        # The next query to send to the Explainer or Coder
        next_query = "Input from USER: " + description + "\n\nPlan:\n" + plan
        
        # Print the plan for debugging and better control
        print("Plan generated:", plan)
        try:
            # Check for specific cases in the response to determine whether to use an agent (or not), if agent then which agent to use
            if (
                "case 2" in low_plan
                or "plan for explanation" in low_plan
                or "plan over which components" in low_plan
            ):
                print("using the Explainer agent")
                
                # Use the Explainer agent to explain the QAOA package
                response = self.explainer.explain(next_query)
            elif "case 0" in low_plan or "not a valid option" in low_plan:
                # If the plan indicates an invalid option, return a direct response
                print("not using an agent, returning response directly")
                response = plan
            else:
                # Use the Coder agent to generate code based on the plan
                print("using the Coder agent")
                response = self.coder.generate_and_test_code(next_query)
            # print("Memory buffer:", self.memory.buffer)
            return response
        except Exception as e:
            print("Error during planning:", e)
            raise

    def __call__(self, description: str) -> str:
        """Call the planner with a description."""
        return self.plan(description)

    def set_context(self):
        """Set or update the context variable."""
        
        # Load context from a file listing valid initial states, problems, and mixers
        filepath = "valid_initialstates_problems_mixers.txt"  
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.context = f.read()
        except FileNotFoundError:
            self.context = "No context available."

# Run planner.py to start the interaction if an interface is not wanted.
if __name__ == "__main__":
    planner = Planner()
    while True:
        query = input("Ask a question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = planner(query)
        print("\nAnswer:")
        print(result)
        print("-" * 40)
