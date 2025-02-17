import json
from pathlib import Path
from typing import Dict, Optional

def get_encoder_alignment(base_path: str, layer: int, encoder_id: str) -> Optional[Dict]:
    """
    Retrieve alignment information for a specific encoder ID from the semantic alignments file.
    
    Args:
        base_path (str): Base path to the alignment files
        layer (int): Layer number
        encoder_id (str): Encoder ID to look up
        
    Returns:
        Optional[Dict]: Alignment information for the specified encoder ID or None if not found
    """
    alignment_file = Path(base_path) / f"layer{layer}" / "semantic_alignments.json"
    
    if not alignment_file.exists():
        print(f"No alignment file found for layer {layer}")
        return None
        
    with open(alignment_file) as f:
        alignment_data = json.load(f)
    
    # Search for the specific encoder ID in alignments
    for alignment in alignment_data["alignments"]:
        if alignment["encoder_id"] == encoder_id:
            return alignment
            
    print(f"No alignment found for encoder ID {encoder_id} in layer {layer}")
    return None

def main():
    # Example usage
    base_path = "coderosetta/cpp_cuda"
    layer = 12  # Change this to the layer you're interested in
    encoder_id = "c16"  # Change this to your desired encoder ID
    
    alignment = get_encoder_alignment(base_path, layer, encoder_id)
    
    if alignment:
        # Create output filename based on layer and encoder_id
        output_file = f"alignment_layer{layer}_{encoder_id}.json"
        
        # Format the data in a more readable structure
        output_data = {
            "encoder_id": alignment['encoder_id'],
            "encoder_cluster": {
                "syntactic_label": alignment['encoder_cluster']['Syntactic Label'],
                "semantic_tags": alignment['encoder_cluster']['Semantic Tags'],
                "tokens": alignment['encoder_cluster']['Unique tokens']
            },
            "matching_decoders": [
                {
                    "decoder_id": match['decoder_id'],
                    "similarity": match['similarity'],
                    "syntactic_label": match['decoder_cluster']['Syntactic Label'],
                    "semantic_tags": match['decoder_cluster']['Semantic Tags'],
                    "tokens": match['decoder_cluster']['Unique tokens']
                }
                for match in alignment['matches']
            ]
        }
        
        # Write to JSON file with proper formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"Alignment data has been written to {output_file}")
    else:
        print("No alignment data found")

if __name__ == "__main__":
    main() 