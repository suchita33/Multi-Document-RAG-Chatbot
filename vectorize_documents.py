from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

loader = DirectoryLoader(
    path="data",            # Folder containing the PDFs
    glob="./*.pdf",         # Specify the file format (PDF)
    loader_cls=UnstructuredFileLoader  # Use UnstructuredFileLoader for extracting text from PDFs
)

pages = loader.load()  # Load all pages
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)
print(len(splits))

vectordb = Chroma.from_texts(
    texts=splits,
    embedding=embeddings,  # Use embed_pages method from embeddings
    persist_directory="vector_db_dir"  # Directory where the vector database will be saved
)

print("pages Vectorized")
