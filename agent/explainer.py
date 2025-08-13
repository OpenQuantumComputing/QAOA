from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import init_chat_model

# ----- Helper imports -----
from saveembedding import SaveEmbedding
from pathlib import Path


class Explainer:
    def __init__(self, memory=None, embedding=None, model="openai:gpt-4.1", temperature=0):
        """
        Initialize the Explainer with the context for QAOA package components.

        Args:
            description (str): The description of the parts to explain.
            model (str): The language model to use.
            temperature (float): The temperature for the language model.
        """
        self.llm = init_chat_model(model, temperature=temperature)
        file_path = self.file_path()
        self.embedding = embedding
        self.context = ""

        if embedding is not None:
            # Making paths where the embeddings are saved and getting the directory of the current file
            current_dir = Path(__file__).resolve().parent
            persist_path = str(current_dir / "embeddings" / "Explainer_embedding")
            cache_path = str(current_dir / "embeddings" / "Explainer_cache")
            

            # Create or extract an embedding that is saved in the persist_path
            make_or_get_embedding = SaveEmbedding(
                dir_paths=file_path,
                collection_name="Explainer_embedding",
                persist_path=persist_path,
                cache_path=cache_path,
            )

            # Get the context, vectorstore, and retriever from the embedding
            self.context = make_or_get_embedding.get_context()
            self.vectorstore = make_or_get_embedding.get_vectorstore()
            self.retriever = make_or_get_embedding.get_retriever()

        if memory is not None:
            self.memory = memory
        else: 
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                return_messages=True,
            )
        self.prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""
        You are an expert on the QAOA package. You get a list over what the USER wants you to explain (they can be for example classes, methods, etc.) and you are going to explain how they work and what attributes, args, and returns they have.
        
        The parts you want to explain are: {question}
        Your context is the documentation strings for the code: {context}
        Here is the chat history: {chat_history}
        
        Make it helpful so that the USER understand the overall meaning of the parts of the package and also how it is used in a code. 
        If you are explaining a method, include the class it belongs to. If you are explaining a class, include its methods and attributes there are any. If you are explaining a variable, include its type and purpose.
        Be concise and structured. 
        If the USER asks for a specific part of the QAOA package, make sure to explain that part in detail.
        Make subtitles for each part you explain, and do NOT use lists or numbered lists.
        NEVER include code snippets in your response.
        Do not include anything the USER has not asked for.
        """,
        )
        # Initialize the chain with the LLM and prompt
        if embedding is not None:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                memory=self.memory,
                retriever=self.retriever,
                combine_docs_chain_kwargs={"prompt": self.prompt},
            )
        else:
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def file_path(self):
        """Set or update the context variable with documentation."""
        script_dir = Path(__file__).resolve().parent.parent

        # Path to the 'qaoa' folder next to it
        folder_path = script_dir / "qaoa"

        # Only get .py files for docstring extraction
        py_file_paths = [
            str(file) for file in folder_path.rglob("*.py") if file.is_file()
        ]

        return py_file_paths

    def explain(self, question):
        """Generate an explanation using the specified context chunk."""
        # Invoke the chain with the question and context
        result = self.chain.invoke({"question": question, "context": self.context})
        if self.embedding is not None:
            return result.get("answer", result)
        else:
            return result.get("text", result)

