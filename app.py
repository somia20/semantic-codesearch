import ast
import os
import logging
import inflection
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import torch
import streamlit as st
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Step 2: Parse the codebase
def parse_python_file(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            source = file.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.error(f"Syntax error in file {file_path}: {e}")
        return []

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            snippet = ast.get_source_segment(source, node)
            docstring = ast.get_docstring(node) or ""
            chunk = {
                "name": node.name,
                "signature": snippet.splitlines()[0],
                "code_type": "Function" if isinstance(node, ast.FunctionDef) else "Class",
                "docstring": docstring,
                "line": node.lineno,
                "line_from": node.lineno,
                "line_to": node.end_lineno,
                "context": {
                    "module": os.path.basename(file_path).replace(".py", ""),
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "struct_name": None if isinstance(node, ast.FunctionDef) else node.name,
                    "snippet": snippet
                }
            }
            chunks.append(chunk)

    logger.info(f"Parsed {len(chunks)} chunks from {file_path}")
    return chunks

def parse_codebase(project_root: str) -> List[Dict[str, Any]]:
    structures = []
    folders_to_skip = {"tests", "docs", "migrations"}
    files_to_skip = {"__init__.py", "setup.py"}

    for folder in os.listdir(project_root):
        folder_path = os.path.join(project_root, folder)
        if os.path.isdir(folder_path) and folder not in folders_to_skip:
            logger.info(f"Processing folder: {folder}")
            for file in os.listdir(folder_path):
                if file.endswith(".py") and file not in files_to_skip:
                    file_path = os.path.join(folder_path, file)
                    chunks = parse_python_file(file_path)
                    structures.extend(chunks)
    logger.info(f"Total chunks parsed: {len(structures)}")
    return structures

# Step 3: Preprocess for NLP
def textify(chunk: Dict[str, Any]) -> str:
    name = inflection.humanize(inflection.underscore(chunk["name"]))
    signature = inflection.humanize(inflection.underscore(chunk["signature"]))
    docstring = f"that does {chunk['docstring']} " if chunk["docstring"] else ""
    context = f"module {chunk['context']['module']} file {chunk['context']['file_name']}"
    if chunk["context"]["struct_name"]:
        struct_name = inflection.humanize(inflection.underscore(chunk["context"]["struct_name"]))
        context = f"defined in struct {struct_name} {context}"

    text_representation = f"{chunk['code_type']} {name} {docstring}defined as {signature} {context}"
    tokens = re.split(r"\W", text_representation)
    tokens = filter(lambda x: x, tokens)
    return " ".join(tokens)

# Step 4: Generate embeddings
def generate_embeddings(structures: List[Dict[str, Any]]) -> tuple:
    text_representations = list(map(textify, structures))
    logger.info(f"Generated {len(text_representations)} text representations")

    nlp_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    code_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code")

    batch_size = 4
    nlp_embeddings = nlp_model.encode(
        text_representations,
        batch_size=batch_size,
        show_progress_bar=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    code_snippets = [structure["context"]["snippet"] for structure in structures]
    code_embeddings = code_model.encode(
        code_snippets,
        batch_size=batch_size,
        show_progress_bar=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return nlp_embeddings, code_embeddings, nlp_model, code_model

# Step 5: Store in Qdrant
def store_in_qdrant(structures: List[Dict[str, Any]], nlp_embeddings, code_embeddings):
    client = QdrantClient(":memory:")
    COLLECTION_NAME = "my-project-codebase"

    client.create_collection(
        COLLECTION_NAME,
        vectors_config={
            "text": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "code": models.VectorParams(size=768, distance=models.Distance.COSINE),
        }
    )

    batch_size = 32
    points = []
    for idx, (text_emb, code_emb, structure) in tqdm(
        enumerate(zip(nlp_embeddings, code_embeddings, structures)), total=len(structures)
    ):
        points.append(
            models.PointStruct(
                id=idx,
                vector={
                    "text": text_emb.tolist(),
                    "code": code_emb.tolist(),
                },
                payload=structure
            )
        )
        if len(points) >= batch_size:
            client.upload_points(COLLECTION_NAME, points=points, wait=True)
            points = []

    if points:
        client.upload_points(COLLECTION_NAME, points=points, wait=True)

    logger.info(f"Total points in collection: {client.count(COLLECTION_NAME).count}")
    return client, COLLECTION_NAME

# Step 6: Search function
def search_codebase(client, collection_name: str, query: str, nlp_model, code_model, limit: int = 5):
    query_text_embedding = nlp_model.encode([query])[0]
    text_hits = client.query_points(
        collection_name,
        query=query_text_embedding.tolist(),
        using="text",
        limit=limit
    ).points

    query_code_embedding = code_model.encode([query])[0]
    code_hits = client.query_points(
        collection_name,
        query=query_code_embedding.tolist(),
        using="code",
        limit=limit
    ).points

    fused_hits = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=query_text_embedding.tolist(), using="text", limit=10),
            models.Prefetch(query=query_code_embedding.tolist(), using="code", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit
    ).points

    text_results = [
        {"Score": hit.score, "Module": hit.payload["context"]["module"], 
         "File Path": hit.payload["context"]["file_path"], "Signature": hit.payload["signature"]}
        for hit in text_hits
    ]
    code_results = [
        {"Score": hit.score, "Module": hit.payload["context"]["module"], 
         "File Path": hit.payload["context"]["file_path"], "Signature": hit.payload["signature"]}
        for hit in code_hits
    ]
    fused_results = [
        {"Score": hit.score, "Module": hit.payload["context"]["module"], 
         "File Path": hit.payload["context"]["file_path"], "Signature": hit.payload["signature"]}
        for hit in fused_hits
    ]

    return text_results, code_results, fused_results

# Streamlit Interface
def main():
    st.title("Code Search with Vector Embeddings")

    # Initialize session state for client and models
    if "client" not in st.session_state:
        project_root = "C:/Users/somia.kumari/botpress/upc"  # Replace with your project path
        structures = parse_codebase(project_root)
        
        # Generate embeddings
        nlp_embeddings, code_embeddings, nlp_model, code_model = generate_embeddings(structures)
        
        # Store in Qdrant
        client, collection_name = store_in_qdrant(structures, nlp_embeddings, code_embeddings)
        
        # Store in session state
        st.session_state["client"] = client
        st.session_state["collection_name"] = collection_name
        st.session_state["nlp_model"] = nlp_model
        st.session_state["code_model"] = code_model
        st.success("Codebase indexed successfully!")

    # Input query
    query = st.text_input("Enter your search query", "how to classify message?")
    limit = st.slider("Number of results", 1, 10, 5)

    if st.button("Search"):
        with st.spinner("Searching..."):
            text_results, code_results, fused_results = search_codebase(
                st.session_state["client"],
                st.session_state["collection_name"],
                query,
                st.session_state["nlp_model"],
                st.session_state["code_model"],
                limit
            )

            # Display results
            st.subheader("Text Embeddings Results")
            if text_results:
                st.dataframe(pd.DataFrame(text_results))
            else:
                st.write("No results found.")

            st.subheader("Code Embeddings Results")
            if code_results:
                st.dataframe(pd.DataFrame(code_results))
            else:
                st.write("No results found.")

            st.subheader("Fused Results.")
            if fused_results:
                st.dataframe(pd.DataFrame(fused_results))
            else:
                st.write("No results found.")

if __name__ == "__main__":
    main()