import argparse
import os
import json
from collections import defaultdict
import google.generativeai as genai
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv
dotenv.load_dotenv()

def read_java_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

def process_cluster_file(file_path, model_dir):
    """Process cluster file and map sentences back to their language sources"""
    clusters_data = defaultdict(lambda: {
        "tokens": set(),
        "sentences": []  # Will store tuples of (sentence, sentence_id, token)
    })
    
    # Load shuffled dataset and source files from model directory
    shuffled_file = os.path.join(model_dir, "shuffled_dataset.txt")
    cpp_file = os.path.join(model_dir, "input.in")
    cuda_file = os.path.join(model_dir, "label.out")
    
    # Load all necessary files
    with open(shuffled_file, 'r', encoding='utf-8') as f:
        shuffled_sentences = [line.strip() for line in f]
    with open(cpp_file, 'r', encoding='utf-8') as f:
        cpp_set = set(line.strip() for line in f)
    with open(cuda_file, 'r', encoding='utf-8') as f:
        cuda_set = set(line.strip() for line in f)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            try:
                parts = stripped_line.split('|||')
                if len(parts) != 5:
                    print(f"Unexpected number of parts ({len(parts)}): {stripped_line}")
                    continue
                    
                token = parts[0].strip()
                sentence_id = int(parts[2])
                cluster_id = parts[-1].strip()
                
                # Get sentence from shuffled dataset
                if 0 <= sentence_id < len(shuffled_sentences):
                    sentence = shuffled_sentences[sentence_id]
                    
                    # Add token to cluster
                    clusters_data[cluster_id]["tokens"].add(token)
                    
                    # Store sentence with its source info
                    sentence_info = {
                        "sentence": sentence,
                        "token": token,
                        "source": "cpp" if sentence in cpp_set else "cuda" if sentence in cuda_set else "unknown"
                    }
                    clusters_data[cluster_id]["sentences"].append(sentence_info)
                    
            except (IndexError, ValueError) as e:
                print(f"Error processing line '{stripped_line}': {str(e)}")
                continue

    # Convert sets to lists for JSON serialization and add language statistics
    result = {}
    for cluster_id, data in clusters_data.items():
        # Count sentences by source
        cpp_count = sum(1 for s in data["sentences"] if s["source"] == "cpp")
        cuda_count = sum(1 for s in data["sentences"] if s["source"] == "cuda")
        unknown_count = sum(1 for s in data["sentences"] if s["source"] == "unknown")
        
        result[cluster_id] = {
            "tokens": list(data["tokens"]),
            "sentences": data["sentences"],
            "stats": {
                "cpp_count": cpp_count,
                "cuda_count": cuda_count,
                "unknown_count": unknown_count,
                "total_sentences": len(data["sentences"])
            }
        }

    # Debug print
    print(f"Total clusters found: {len(clusters_data)}")
    print(f"Processing only first 5 clusters for testing")
    
    # Limit to first 5 clusters
    limited_clusters = dict(list(clusters_data.items())[:5])
    return limited_clusters

def label_clusters(clusters_data):
    # Initialize Gemini API
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    
    generation_config = {
        "temperature": 0.3,
        "top_p": 0.4,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",  # Explicitly request JSON response
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        system_instruction="You are a C++ and CUDA Software Developer experienced in analyzing code. Always respond with valid JSON."
    )

    final_output = []
    max_retries = 10
    retry_delay = 2  # seconds

    # Updated safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Print total clusters to process
    print(f"Processing {len(clusters_data)} clusters (limited to 5 for testing)")
    
    # Only process clusters that exist in the data
    for index, (cluster_id, cluster_info) in enumerate(clusters_data.items()):
        print(f"Processing cluster {cluster_id} ({index + 1}/{len(clusters_data)})")
        
        # Rate limiting
        if index > 0 and index % 2000 == 0:
            print("Pausing for 60 seconds due to rate limit...")
            time.sleep(61)

        tokens_list = cluster_info["tokens"]
        sentences = cluster_info["sentences"]

        # Separate C++ and CUDA sentences
        cpp_sentences = [s for s in sentences if s['source'] == 'cpp']
        cuda_sentences = [s for s in sentences if s['source'] == 'cuda']

        # Separate tokens based on which sentences they appear in
        cpp_tokens = {token for token in tokens_list if any(token == s['token'] for s in cpp_sentences)}
        cuda_tokens = {token for token in tokens_list if any(token == s['token'] for s in cuda_sentences)}

        # Updated prompt in the user_message
        user_message = f"""
        You are given a concept cluster of tokens which contains tokens from C++ and CUDA functions as a 
        part of a study to identify latent concepts learned in contextualized representations of a neural network. 
        The concept cluster was obtained by clustering (KMeans) the neuron activations from a model with Masked Language 
        Modeling (MLM) as its objective.
        
        NOTE: Some clusters might have only C++ or CUDA tokens and context sentences.
        
        Guidelines for Analysis:
        1. **Lexical Patterns**: Analyze why these tokens might have been clustered together. Look for:
           - Lexical/syntactic patterns (e.g., similar syntax roles, token positions)
           - Common programming constructs they represent
           - Any patterns in how they're used in both C++ and CUDA contexts
        
        2. **Semantic Tags**:
           - Identify the broader purpose these tokens serve in the code contexts
           - Note any domain-specific usage patterns
           - Consider the programming concepts they represent
           - For each semantic tag, give the tokens that it is associated with and reason why they are associated with the tag.
           
        3. **Functional Equivalence**:
           - Analyze if the context sentences represent similar functionality across C++ and CUDA
           - Note any performance-related patterns 
        
        4. **Common Semantic Patterns**:
           - Identify any common semantic patterns or concepts that these context sentences have in common.
           - Dont focus on tokens, focus on the semantic patterns and concepts in the context sentences.
           - Give a detailed description of the common semantic patterns

        ## Analyze this cluster now with the following tokens and context sentences:

        C++ Tokens: {', '.join(cpp_tokens)}
        C++ Context Sentences:
        {chr(10).join([f"{i + 1}. {s['sentence']}" for i, s in enumerate(cpp_sentences)])}

        CUDA Tokens: {', '.join(cuda_tokens)}
        CUDA Context Sentences:
        {chr(10).join([f"{i + 1}. {s['sentence']}" for i, s in enumerate(cuda_sentences)])}
        
        Respond with this exact JSON structure:
        {{
            "lexical_patterns": "string",
            "semantic_tags": ["string", "string", "string", "string", "string"],
            "functional_equivalence": "string",
            "semantic_description": "string"
        }}"""
    
        success = False
        attempts = 0
        
        while not success and attempts < max_retries:
            try:
                response = model.generate_content(
                    user_message, 
                    safety_settings=safety_settings,
                    generation_config=generation_config
                )
                

                # Simplified response cleaning - just get the raw text
                cleaned_response = response.text.strip()
                
                # Try to parse as JSON directly first
                try:
                    response_json = json.loads(cleaned_response)
                except json.JSONDecodeError as je:
                    print(f"JSON decode error: {je}")
                    # If direct parsing fails, try to extract JSON from markdown
                    if '```json' in cleaned_response:
                        json_content = cleaned_response.split('```json')[1].split('```')[0].strip()
                        print(f"Extracted JSON content: {json_content}")
                        response_json = json.loads(json_content)
                    else:
                        raise ValueError("Response is not valid JSON")

                # Validate response structure with detailed error messages
                required_fields = [ 'lexical_patterns', 'functional_equivalence', 
                                 'semantic_tags', 'semantic_description']
                
                missing_fields = [field for field in required_fields if field not in response_json]
                if missing_fields:
                    raise ValueError(f"Response missing required fields: {missing_fields}")
                
                if not isinstance(response_json['semantic_tags'], list):
                    raise ValueError("semantic_tags must be a list")
                
                if len(response_json['semantic_tags']) != 5:
                    raise ValueError(f"semantic_tags must have exactly 5 elements, got {len(response_json['semantic_tags'])}")
                
                # If we get here, all validations passed
                final_output.append({
                    "c" + cluster_id: {
                        "unique_tokens": list(cluster_info["tokens"]),
                        **response_json
                    }
                })
                
                success = True
                print(f"Successfully processed cluster {cluster_id}")
                
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed: {str(e)}")
                if attempts < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to process cluster {cluster_id} after {max_retries} attempts")
                    # Add fallback response
                    final_output.append({
                        "c" + cluster_id: {
                            "unique_tokens": list(cluster_info["tokens"]),
                            "lexical_patterns": "Processing failed",
                            "functional_equivalence": "Processing failed",
                            "semantic_tags": ["error1", "error2", "error3", "error4", "error5"],
                            "semantic_description": f"Failed to process after {max_retries} attempts"
                        }
                    })
                    break

    return final_output

def process_single_cluster_file(directory, filename, model_dir):
    """Helper function to process a single cluster file and save its results"""
    cluster_file = os.path.join(directory, filename)
    print(f"Processing mixed cluster file: {cluster_file}")
    
    # Add debug prints
    print(f"Reading clusters from: {cluster_file}")
    clusters_data = process_cluster_file(cluster_file, model_dir)
    print(f"Found {len(clusters_data)} clusters")
    
    final_output = label_clusters(clusters_data)
    print(f"Generated labels for {len(final_output)} clusters")
    
    json_output_path = os.path.join(directory, f'mixed_gemini_labels.json')
    with open(json_output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved JSON to {json_output_path}")

def main(args):
    layer_dirs = sorted(d for d in os.listdir(args.model_dir) 
                       if os.path.isdir(os.path.join(args.model_dir, d)) 
                       and d.startswith("layer")
                       and args.start_layer <= int(d.replace("layer", "")) <= args.end_layer)

    if not layer_dirs:
        print(f"No layer directories found in {args.model_dir}")
        return

    for layer_dir in layer_dirs:
        print(f"\nProcessing {layer_dir}")
        layer_path = os.path.join(args.model_dir, layer_dir)

        # Look for mixed cluster files
        cluster_files = [f for f in os.listdir(layer_path) 
                        if f.startswith("clusters-kmeans") and f.endswith(".txt")]
        
        if cluster_files:
            print(f"Processing file: {cluster_files[0]}")
            process_single_cluster_file(layer_path, cluster_files[0], args.model_dir)
        else:
            print(f"No mixed cluster files found in {layer_path}")

def test_cluster_loading(model_dir, layer_num):
    """Test function to verify cluster loading and token/sentence separation"""
    layer_path = os.path.join(model_dir, f"layer{layer_num}")
    
    # Find the cluster file
    cluster_files = [f for f in os.listdir(layer_path) 
                    if f.startswith("clusters-kmeans") and f.endswith(".txt")]
    
    if not cluster_files:
        print("No cluster files found!")
        return
        
    cluster_file = os.path.join(layer_path, cluster_files[0])
    print(f"\nProcessing cluster file: {cluster_file}")
    
    # Process the cluster file
    clusters_data = process_cluster_file(cluster_file, model_dir)
    
    # Test a few clusters
    for cluster_id in list(clusters_data.keys())[:1]:  # Test first 3 clusters
        print(f"\n{'='*80}")
        print(f"Analyzing Cluster {cluster_id}:")
        print('='*80)
        cluster = clusters_data[cluster_id]
        
        # Get sentences by type
        cpp_sentences = [s for s in cluster["sentences"] if s['source'] == 'cpp']
        cuda_sentences = [s for s in cluster["sentences"] if s['source'] == 'cuda']
        
        # Get tokens by type
        tokens_list = cluster["tokens"]
        cpp_tokens = {token for token in tokens_list if any(token == s['token'] for s in cpp_sentences)}
        cuda_tokens = {token for token in tokens_list if any(token == s['token'] for s in cuda_sentences)}
        
        print(f"\nStatistics:")
        print(f"- Total tokens: {len(tokens_list)}")
        print(f"- C++ tokens: {len(cpp_tokens)}")
        print(f"- CUDA tokens: {len(cuda_tokens)}")
        print(f"- C++ sentences: {len(cpp_sentences)}")
        print(f"- CUDA sentences: {len(cuda_sentences)}")
        
        print("\nC++ Tokens:", ', '.join(cpp_tokens))
        print("CUDA Tokens:", ', '.join(cuda_tokens))
        
        print("\nC++ Sentences:")
        if cpp_sentences:
            for i, s in enumerate(cpp_sentences, 1):
                print(f"{i}. Token: '{s['token']}' | Sentence: {s['sentence']}")
        else:
            print("No C++ sentences found")
            
        print("\nCUDA Sentences:")
        if cuda_sentences:
            for i, s in enumerate(cuda_sentences, 1):
                print(f"{i}. Token: '{s['token']}' | Sentence: {s['sentence']}")
        else:
            print("No CUDA sentences found")
        
        print('\n' + '-'*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and label clusters")
    parser.add_argument("--model-dir", required=True, help="Path to the model directory containing layer folders")
    parser.add_argument("--start-layer", type=int, required=True, help="Layer number to start from")
    parser.add_argument("--end-layer", type=int, required=True, help="Layer number to end at")
    parser.add_argument("--test-only", action="store_true", help="Only run the loading test")
    args = parser.parse_args()

    if args.test_only:
        test_cluster_loading(args.model_dir, args.start_layer)
    else:
        main(args)