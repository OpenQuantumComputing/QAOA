# to make the code splitter
from pathlib import Path
from typing import List
import json

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import ast
from langchain.schema import Document

import os
from langchain_chroma import Chroma

class SaveEmbedding:
    def __init__(
        self,
        dir_paths,
        collection_name,
        persist_path,
        cache_path,
        type_of_receiver="mmr"
    ):
        self.dir_paths = dir_paths
        self.collection_name = collection_name
        self.persist_path = persist_path
        self.cache_path = cache_path
        self.context = None
        self.split_docs = None
        self.vectorstore = None
        self.retriever = None

        self.load_context()
        self.split()
        self.create_retriever(type_of_receiver)
        
    def load_python_files(self, repo_path):
        """
        Load all Python files from the specified repository path.
        """
        loader_py = DirectoryLoader(repo_path, glob="**/*.py")
        docs_py = loader_py.load()
        return docs_py

    def load_notebook(self, path: Path) -> str:
        """Load Jupyter notebook content."""
        with open(path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
        content = []
        for cell in notebook["cells"]:
            if cell["cell_type"] in ["markdown", "code"]:
                cell_content = "\n".join(cell["source"])
                content.append(cell_content)
                # print(f"Loaded cell content:\n{cell_content}\n{'-'*50}")
        return "\n\n".join(content)

    def load_python_script(self, path: Path) -> str:
        """Load Python script content."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_text_file(self, path: Path) -> str:
        """Load text or markdown file content."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_class_docstrings_from_string(self, code: str) -> str:
        """Extract class-level docstrings from a Python source string."""
        extracted_docs = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        extracted_docs.append(docstring.strip())
        except SyntaxError:
            pass  # Skip files that fail to parse
        return "\n\n".join(extracted_docs)

    def load_context(self):
        """Loads and processes documents from the specified paths."""
        context = []
        for path in self.dir_paths:
            path = Path(path)
            if not path.exists():
                print(f"File not found - {path}. Skipping.")
                continue
            if path.suffix == ".ipynb":
                context.append(self.load_notebook(path))
            elif path.suffix in [".txt", ".md"]:
                context.append(self.load_text_file(path))
            elif path.suffix == ".py":
                py_str = self.load_python_script(path)
                docstring_str = self.extract_class_docstrings_from_string(py_str)
                context.append(docstring_str)
            else:
                print(f"Unsupported file type - {path.suffix}. Skipping.")

        print(f"\nLoaded {len(context)} files.")
        self.context = context

    def split(self):
        print(f"Processing {len(self.context)} raw documents.")
        try:
            docs = [
                Document(page_content=context_str.strip())
                for context_str in self.context
            ]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=200
            )
            split_docs = splitter.split_documents(docs)
            print(f"Generated {len(split_docs)} split documents.")

            if not split_docs:
                print(
                    "No documents to process after splitting. Skipping vectorstore creation."
                )
            
            self.split_docs = split_docs
        except Exception as e:
            print(f"Error processing documents: {str(e)}")

    def create_retriever(self, type_of_receiver):
        if os.path.exists(self.persist_path):
            print(f"Loading existing vectorstore from {self.persist_path}")
            self.vectorstore = Chroma(
                embedding_function=OpenAIEmbeddings(),
                persist_directory=self.persist_path,
                collection_name=self.collection_name,
            )
        else:
            # print(self.split_docs)
            print(f"Creating new vectorstore at {self.persist_path}")
            self.vectorstore = Chroma.from_documents(
                documents=self.split_docs,
                embedding=OpenAIEmbeddings(),
                persist_directory=self.persist_path,
                collection_name=self.collection_name,
            )

        if type_of_receiver == "mmr":
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.9},
            )

        else:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 8}
            )  # TODO check out if this is what we want

    def get_retriever(self):
        """Get the retriever."""
        return self.retriever

    def get_vectorstore(self):
        """Get the vectorstore."""
        return self.vectorstore

    def get_context(self):
        """Get the context."""
        return self.context