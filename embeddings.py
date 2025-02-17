import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def create_cluster_text(self, cluster: Dict) -> str:
        """Create a text representation of a cluster for embedding."""
        text = f"""
        Syntactic Label: {cluster['Syntactic Label']}
        Semantic Tags: {', '.join(cluster['Semantic Tags'])}
        Tokens: {', '.join(cluster['Unique tokens'])}
        Description: {cluster.get('Description', '')}
        """
        return text.strip()

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a given text."""
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, truncation=True, 
                                  max_length=512, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings[0]

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

class ClusterAligner:
    def __init__(self, base_path: str, layer: int):
        self.base_path = Path(base_path)
        self.layer = layer
        self.embedder = ClusterEmbedder()
        self.similarity_threshold = 0.82  # Good balance between precision/recall
        
    def load_clusters(self) -> Tuple[List[Dict], List[Dict]]:
        """Load raw encoder and decoder clusters."""
        # Load encoder clusters
        encoder_file = self.base_path / f"layer{self.layer}" / f"encoder_gemini_labels.json"
        with open(encoder_file) as f:
            encoder_clusters = json.load(f)
            
        # Load decoder clusters
        decoder_file = self.base_path / f"layer{self.layer}" / f"decoder_gemini_labels.json" 
        with open(decoder_file) as f:
            decoder_clusters = json.load(f)
                
        return encoder_clusters, decoder_clusters

    def align_clusters(self) -> List[Dict]:
        """Align encoder and decoder clusters based on semantic similarity."""
        encoder_clusters, decoder_clusters = self.load_clusters()
        alignments = []

        # Pre-compute all decoder embeddings
        logger.info("Pre-computing decoder embeddings...")
        decoder_embeddings = {}
        for dec_cluster_dict in tqdm(decoder_clusters):
            dec_id = list(dec_cluster_dict.keys())[0]
            dec_cluster = dec_cluster_dict[dec_id]
            dec_text = self.embedder.create_cluster_text(dec_cluster)
            decoder_embeddings[dec_id] = {
                'embedding': self.embedder.get_embedding(dec_text),
                'cluster': dec_cluster
            }

        # Process encoder clusters and find matches
        logger.info(f"Processing {len(encoder_clusters)} encoder clusters...")
        for enc_cluster_dict in tqdm(encoder_clusters):
            enc_id = list(enc_cluster_dict.keys())[0]
            enc_cluster = enc_cluster_dict[enc_id]
            
            enc_text = self.embedder.create_cluster_text(enc_cluster)
            enc_embedding = self.embedder.get_embedding(enc_text)
            
            matches = []
            for dec_id, dec_data in decoder_embeddings.items():
                similarity = self.embedder.compute_similarity(enc_embedding, dec_data['embedding'])
                
                if similarity >= self.similarity_threshold:
                    matches.append({
                        "decoder_id": dec_id,
                        "decoder_cluster": dec_data['cluster'],
                        "similarity": float(similarity)
                    })
            
            alignments.append({
                "encoder_id": enc_id,
                "encoder_cluster": enc_cluster,
                "matches": sorted(matches, key=lambda x: x["similarity"], reverse=True)
            })

        return alignments

    def save_alignments(self, alignments: List[Dict]):
        """Save alignment results to file."""
        output_file = self.base_path / f"layer{self.layer}" / "semantic_alignments.json"
        with open(output_file, "w") as f:
            json.dump({
                "layer": self.layer,
                "alignments": alignments,
                "metadata": {
                    "similarity_threshold": self.similarity_threshold,
                    "model": self.embedder.model.name_or_path,
                    "total_alignments": len(alignments)
                }
            }, f, indent=2)
        
        logger.info(f"Saved alignments to {output_file}")

def main():
    base_path = "coderosetta/cpp_cuda"
    # Process all 13 layers
    for layer in [12]:
        logger.info(f"Processing layer {layer}")
        aligner = ClusterAligner(base_path, layer)
        
        # Check if output file already exists
        output_file = Path(base_path) / f"layer{layer}" / "semantic_alignments.json"
        if output_file.exists():
            logger.info(f"Output file already exists for layer {layer}, skipping...")
            continue
            
        alignments = aligner.align_clusters()
        aligner.save_alignments(alignments)
        logger.info(f"Found {len(alignments)} alignments for layer {layer}")

if __name__ == "__main__":
    main()
