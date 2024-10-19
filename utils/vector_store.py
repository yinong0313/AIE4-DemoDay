from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters.base import TextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
import tiktoken

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from abc import ABC, abstractmethod
import re

import pymupdf4llm

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text)
    return len(tokens)

def replace_newlines(text):
    # Replace consecutive newlines (two or more) with the same number of <br>
    text = re.sub(r'\n{2,}', '\n\n', text)
    # Replace single newlines with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Ensure there is a blank line before headings
    text = re.sub(r'([^\n])\n(#+)', r'\1\n\n\2', text)
    text = re.sub(r'([^\n|#])(#+)', r'\1\n\n\2', text)
    
    return text

def get_markdown_documents(path, **kwargs):
    md = pymupdf4llm.to_markdown(path, force_text=True)
    md = replace_newlines(md)
    
    chunk_size = kwargs.get('chunk_size')
    chunk_overlap = kwargs.get('chunk_overlap')
    
    markdown_splitter = MarkdownTextSplitter(chunk_size = chunk_size,
                                        chunk_overlap = chunk_overlap,
                                        length_function = tiktoken_len,
                                        )
    documents = markdown_splitter.create_documents([md])
    return documents

class Chunking(ABC):
    
    """Abstract method for basic and advanced chunking strategy"""
    
    def __init__(self, file_path: str, loader: BaseLoader, splitter: TextSplitter):
        self.file_path = file_path
        self.loader = loader
        self.splitter = splitter
        
    @abstractmethod
    def process_documents(self):
        pass


class ChunkDocument(Chunking):
    '''
    Choose your document loader and text splitter and chunk the document
    '''
    def __init__(self, file_path: str, loader: BaseLoader, splitter: TextSplitter):
        super().__init__(file_path, loader, splitter)
    
    def process_documents(self, **kwargs):
        '''
        Read a single document and chunk it
        '''
        docs = self.loader(self.file_path).load()
        chunks = self.splitter(**kwargs).split_documents(docs)
        print(len(chunks))
        return chunks


def get_vector_store(documents: List, embedding_model: HuggingFaceEmbeddings, emb_dim: int, collection_name: str) -> QdrantVectorStore:
    '''
    Return a qdrant vector score retriever
    '''
    
    qdrant_client = QdrantClient(":memory:")

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE)
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding_model
        )

    vector_store.add_documents(documents)

    return vector_store