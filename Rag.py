import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Union
import uuid

class VideoFrameVectorDB:
    def __init__(self, collection_name: str, persistent_path: Optional[str] = None):
        """
        Initialize the Chroma vector database and CLIP model for video frame embeddings.
        
        Args:
            collection_name (str): Name of the Chroma collection.
            persistent_path (Optional[str]): Path for persistent storage, None for in-memory.
        """
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persistent_path) if persistent_path else chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        
        # Load CLIP model for image embeddings
        self.model = SentenceTransformer('clip-ViT-B-32')

    def generate_embedding(self, frame: Union[Image.Image, str]) -> List[float]:
        """
        Generate an embedding for a single video frame or image file.
        
        Args:
            frame (Union[Image.Image, str]): PIL Image object or path to image file.
        
        Returns:
            List[float]: 512-dimensional embedding vector.
        """
        if isinstance(frame, str):
            frame = Image.open(frame)
        elif not isinstance(frame, Image.Image):
            raise ValueError("Frame must be a PIL Image object or a file path.")
        
        # Convert frame to RGB if needed (CLIP expects RGB)
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')
        
        # Generate embedding
        embedding = self.model.encode(frame).tolist()
        return embedding

    def insert_frame(self, frame: Union[Image.Image, str], label: str, id: Optional[str] = None) -> str:
        """
        Insert a single video frame embedding with a label into the database.
        
        Args:
            frame (Union[Image.Image, str]): PIL Image object or path to image file.
            label (str): Label associated with the frame (e.g., 'person', 'car').
            id (Optional[str]): Unique ID for the embedding, auto-generated if None.
        
        Returns:
            str: ID of the inserted embedding.
        """
        # Generate embedding
        embedding = self.generate_embedding(frame)
        
        # Generate ID if not provided
        id = id if id else str(uuid.uuid4())
        
        # Store embedding with label as metadata
        metadata = {"label": label}
        if isinstance(frame, str):
            metadata["path"] = frame
        
        self.collection.add(
            embeddings=[embedding],
            ids=[id],
            metadatas=[metadata]
        )
        
        #save the frame to a file 
        frame.save(f"rag_images/{id}.jpg")
        print(f"Saved frame to rag_images/{id}.jpg") 
        
        return id

    def search_similar_frames(self, query_frame: Union[Image.Image, str], n_results: int = 3, label_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for similar frames based on a query frame.
        
        Args:
            query_frame (Union[Image.Image, str]): PIL Image object or path to query image.
            n_results (int): Number of results to return.
            label_filter (Optional[str]): Filter results by label (e.g., 'person').
        
        Returns:
            List[Dict]: List of results with IDs, labels, and similarity scores (and paths if available).
        """
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query_frame)
        
        # Perform similarity search
        query_params = {"n_results": n_results}
        if label_filter:
            query_params["where"] = {"label": label_filter}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            **query_params
        )
        

       
        # Format results
        formatted_results = []
        for id, distance, metadata in zip(results["ids"][0], results["distances"][0], results["metadatas"][0]):
            result = {
                "id": id,
                "label": metadata["label"],
                "distance": distance
            }
            if "path" in metadata:
                result["path"] = metadata["path"]
            formatted_results.append(result)
     
        return formatted_results