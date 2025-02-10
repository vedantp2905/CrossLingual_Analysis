import streamlit as st
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
from collections import defaultdict
from supabase import create_client
from dotenv import load_dotenv
from typing import List
import io
import pandas as pd
import time

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

def load_sentences(model_dir: str, component: str):
    """Load sentences from input.in or label.out based on component"""
    file_name = "input.in" if component == "encoder" else "label.out"
    file_path = os.path.join(model_dir, file_name)
    
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def create_wordcloud(tokens):
    """Create and return a word cloud from tokens"""
    if not tokens:
        return None
        
    # Create frequency dict
    freq_dict = {token: 1 for token in tokens}
    
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100
    ).generate_from_frequencies(freq_dict)
    
    return wc

def get_language_statistics(sentences: List[dict], model_dir: str) -> dict:
    """
    Analyze sentences to determine if they come from C++ or CUDA sources
    Returns statistics about language distribution
    """
    # Load original source files
    cpp_sentences = set()
    cuda_sentences = set()
    
    try:
        with open(os.path.join(model_dir, "input.in"), 'r', encoding='utf-8') as f:
            cpp_sentences = set(line.strip() for line in f)
        with open(os.path.join(model_dir, "label.out"), 'r', encoding='utf-8') as f:
            cuda_sentences = set(line.strip() for line in f)
    except FileNotFoundError:
        return None
        
    # Initialize counters
    stats = {
        "cpp_count": 0,
        "cuda_count": 0,
        "mixed_count": 0,
        "unknown_count": 0,
        "total_tokens": len(sentences),
        "cpp_sentences": [],
        "cuda_sentences": [],
        "mixed_sentences": [],
        "unknown_sentences": []
    }
    
    # Track unique tokens per category
    unique_tokens = {
        "cpp": set(),
        "cuda": set(), 
        "mixed": set(),
        "unknown": set()
    }
    
    # Analyze each sentence
    for sent_info in sentences:
        sentence = sent_info["sentence"].strip()
        token = sent_info["token"]
        
        in_cpp = sentence in cpp_sentences
        in_cuda = sentence in cuda_sentences
        
        if in_cpp and in_cuda:
            stats["mixed_count"] += 1
            stats["mixed_sentences"].append((token, sentence))
            unique_tokens["mixed"].add(token)
        elif in_cpp:
            stats["cpp_count"] += 1
            stats["cpp_sentences"].append((token, sentence))
            unique_tokens["cpp"].add(token)
        elif in_cuda:
            stats["cuda_count"] += 1
            stats["cuda_sentences"].append((token, sentence))
            unique_tokens["cuda"].add(token)
        else:
            stats["unknown_count"] += 1
            stats["unknown_sentences"].append((token, sentence))
            unique_tokens["unknown"].add(token)
    
    # Add unique token counts to stats
    stats.update({
        "unique_cpp_tokens": len(unique_tokens["cpp"]),
        "unique_cuda_tokens": len(unique_tokens["cuda"]),
        "unique_mixed_tokens": len(unique_tokens["mixed"]),
        "unique_unknown_tokens": len(unique_tokens["unknown"])
    })
            
    return stats

def display_language_statistics(stats: dict):
    """Display language statistics in Streamlit"""
    if not stats:
        st.warning("Could not load source files for language statistics")
        return
        
    st.write("### Language Distribution Statistics")
    
    total = stats["total_tokens"]
    
    # Create metrics for token counts
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("C++ Tokens", f"{stats['cpp_count']} ({(stats['cpp_count']/total)*100:.1f}%)")
        st.metric("Unique C++ Tokens", stats['unique_cpp_tokens'])
    with col2:
        st.metric("CUDA Tokens", f"{stats['cuda_count']} ({(stats['cuda_count']/total)*100:.1f}%)")
        st.metric("Unique CUDA Tokens", stats['unique_cuda_tokens'])
    with col3:
        st.metric("Mixed Tokens", f"{stats['mixed_count']} ({(stats['mixed_count']/total)*100:.1f}%)")
        st.metric("Unique Mixed Tokens", stats['unique_mixed_tokens'])
    with col4:
        st.metric("Unknown", f"{stats['unknown_count']} ({(stats['unknown_count']/total)*100:.1f}%)")
        st.metric("Unique Unknown Tokens", stats['unique_unknown_tokens'])
    
    # Create detailed view with tabs
    st.write("### Detailed Token Distribution")
    tab1, tab2, tab3, tab4 = st.tabs(["C++", "CUDA", "Mixed", "Unknown"])
    
    def highlight_exact_token(sentence: str, token: str) -> str:
        """Highlight exact token matches only"""
        words = sentence.split()
        highlighted_words = [f"<span style='color: red; font-weight: bold;'>{word}</span>" if word == token else word for word in words]
        return ' '.join(highlighted_words)
    
    with tab1:
        if stats["cpp_sentences"]:
            for token, sentence in stats["cpp_sentences"]:
                html = f"""
                <div style='font-family: monospace; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 5px;'>
                    <div style='margin-bottom: 5px;'>
                        {highlight_exact_token(sentence, token)}
                    </div>
                    <div style='color: #666; font-size: 0.9em;'>Token: <code>{token}</code></div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.write("No C++ tokens found")
            
    with tab2:
        if stats["cuda_sentences"]:
            for token, sentence in stats["cuda_sentences"]:
                html = f"""
                <div style='font-family: monospace; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 5px;'>
                    <div style='margin-bottom: 5px;'>
                        {highlight_exact_token(sentence, token)}
                    </div>
                    <div style='color: #666; font-size: 0.9em;'>Token: <code>{token}</code></div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.write("No CUDA tokens found")
            
    with tab3:
        if stats["mixed_sentences"]:
            for token, sentence in stats["mixed_sentences"]:
                html = f"""
                <div style='font-family: monospace; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 5px;'>
                    <div style='margin-bottom: 5px;'>
                        {highlight_exact_token(sentence, token)}
                    </div>
                    <div style='color: #666; font-size: 0.9em;'>Token: <code>{token}</code></div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.write("No mixed tokens found")
            
    with tab4:
        if stats["unknown_sentences"]:
            for token, sentence in stats["unknown_sentences"]:
                html = f"""
                <div style='font-family: monospace; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 5px;'>
                    <div style='margin-bottom: 5px;'>
                        {highlight_exact_token(sentence, token)}
                    </div>
                    <div style='color: #666; font-size: 0.9em;'>Token: <code>{token}</code></div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.write("No unknown tokens found")

def display_cluster_info(cluster_data, model_pair: str, layer_number: int, cluster_id: str, sentences=None):
    """Display cluster information including word cloud, metadata and sentences"""
    # Get model name from model_pair
    model = model_pair.split('/')[0]
    
    if model in ["coderosetta_mlm_mixed", "coderosetta_aer_mixed"]:
        # For MLM mixed model and AER mixed model, show statistics and sentences
        if sentences:
            # Create word cloud from unique tokens if available
            if isinstance(sentences, dict) and "unique_tokens" in sentences:
                st.write("### Word Cloud")
                wc = create_wordcloud(sentences["unique_tokens"])
                if wc:
                    # Reduced figure size for mixed models
                    fig = plt.figure(figsize=(10, 5))  # Smaller size (was 10, 5)
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Use the sentences list for statistics
                stats = get_language_statistics(sentences["sentences"], os.path.join(model, model_pair.split('/')[1]))
            else:
                stats = get_language_statistics(sentences, os.path.join(model, model_pair.split('/')[1]))
            
            display_language_statistics(stats)
            return

    # Store model_pair in session state to persist across reruns
    if 'model_pair' not in st.session_state:
        st.session_state.model_pair = model_pair
    
    # Initialize session state for cluster navigation if not exists
    if 'current_cluster_index' not in st.session_state:
        st.session_state.current_cluster_index = 0
    
    # Word cloud and metadata in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tokens = cluster_data.get("Unique tokens", [])
        wc = create_wordcloud(tokens)
        if wc:
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
    with col2:
        st.write("### Metadata")
        st.write(f"**Syntactic Label:** {cluster_data.get('Syntactic Label', 'N/A')}")
        st.write("**Semantic Tags:**")
        for tag in cluster_data.get('Semantic Tags', []):
            st.write(f"- {tag}")
        st.write(f"**Description:** {cluster_data.get('Description', 'N/A')}")
        
        # Add evaluation section
        st.write("---")
        st.write("### Evaluation")
        
        # Syntactic accuracy
        syntactic_accuracy = st.radio(
            "Is the syntactic label accurate?",
            ["Accurate", "Semi-accurate", "Not accurate"],
            key="syntactic_radio"
        )
        
        # Semantic accuracy
        semantic_accuracy = st.radio(
            "Are the semantic tags accurate?",
            ["Accurate", "Semi-accurate", "Not accurate"],
            key="semantic_radio"
        )
        
        # Notes field
        if syntactic_accuracy == "Not accurate" or semantic_accuracy == "Not accurate":
            notes = st.text_area(
                "Please provide notes explaining why the labels are not accurate:",
                key="notes",
                help="Required for 'Not accurate' selections"
            )
        else:
            notes = st.text_area(
                "Additional notes (optional):",
                key="notes_optional"
            )
        
        # Submit button
        if st.button("Submit Evaluation"):
            if (syntactic_accuracy == "Not accurate" or semantic_accuracy == "Not accurate") and not notes.strip():
                st.error("Please provide notes explaining why the labels are not accurate.")
            else:
                evaluation_data = {
                    "syntactic_accuracy": syntactic_accuracy,
                    "semantic_accuracy": semantic_accuracy,
                    "notes": notes
                }
                
                model = model_pair.split('/')[0]  # Get just t5 or coderosetta
                language_pair = model_pair.split('/')[1]  # Get the language pair
                
                if save_cluster_evaluation(
                    model=model,
                    language_pair=language_pair,
                    layer_number=layer_number,
                    cluster_id=cluster_id,
                    evaluation_data=evaluation_data
                ):
                    st.success("Evaluation submitted successfully!")
                    # Add JavaScript to scroll to top before rerun
                    js = '''
                        <script>
                            window.scrollTo(0, 0);
                            var elements = window.parent.document.getElementsByTagName('iframe');
                            for (var i = 0; i < elements.length; i++) {
                                elements[i].contentWindow.scrollTo(0, 0);
                            }
                        </script>
                    '''
                    st.markdown(js, unsafe_allow_html=True)
                    st.session_state.current_cluster_index += 1
                    st.rerun()

    # Display context sentences
    if sentences:
        st.write("---")
        st.write("### Context Sentences")
        
        with st.container():
            for sent_info in sentences:
                tokens = sent_info["sentence"].split()
                html = create_sentence_html(tokens, sent_info)
                st.markdown(html, unsafe_allow_html=True)

def load_cluster_sentences(model_dir: str, layer: int, component: str):
    """Load sentences and their indices from cluster file"""
    # Special handling for mixed clusters
    if component == "mixed":
        cluster_file = os.path.join(model_dir, f"layer{layer}", f"clusters-kmeans-500.txt")
        sentence_file = os.path.join(model_dir, "shuffled_dataset.txt")
        
        # Initialize dict to store unique tokens per cluster
        unique_tokens_per_cluster = defaultdict(set)
        
        # Load all sentences first
        with open(sentence_file, 'r', encoding='utf-8') as f:
            all_sentences = [line.strip() for line in f]
        
        # Process cluster file to get sentence mappings
        cluster_sentences = defaultdict(list)
        
        with open(cluster_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|||')
                if len(parts) == 5:  # Expected format: token|||other|||sent_id|||token_idx|||cluster_id
                    token = parts[0].strip()
                    sentence_id = int(parts[2])
                    token_idx = int(parts[3])
                    cluster_id = parts[4].strip()
                    cluster_key = f"c{cluster_id}"
                    
                    if 0 <= sentence_id < len(all_sentences):
                        cluster_sentences[cluster_key].append({
                            "sentence": all_sentences[sentence_id],
                            "token": token,
                            "token_idx": token_idx
                        })
                        # Add token to unique tokens set for this cluster
                        unique_tokens_per_cluster[cluster_key].add(token)
        
        # Convert unique tokens sets to lists and add to cluster_sentences
        for cluster_id in cluster_sentences:
            cluster_sentences[cluster_id] = {
                "sentences": cluster_sentences[cluster_id],
                "unique_tokens": list(unique_tokens_per_cluster[cluster_id])
            }
        
        return cluster_sentences
    else:
        # Original logic for encoder/decoder
        cluster_file = os.path.join(model_dir, f"layer{layer}", f"{component}-clusters-kmeans-500.txt")
        sentence_file = os.path.join(model_dir, "input.in" if component == "encoder" else "label.out")
    
    # Load all sentences first
    with open(sentence_file, 'r', encoding='utf-8') as f:
        all_sentences = [line.strip() for line in f]
    
    # Process cluster file to get sentence mappings
    cluster_sentences = defaultdict(list)
    
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|||')
            if len(parts) == 5:  # Expected format: token|||other|||sent_id|||token_idx|||cluster_id
                token = parts[0].strip()
                sentence_id = int(parts[2])
                token_idx = int(parts[3])
                cluster_id = parts[4].strip()
                
                if 0 <= sentence_id < len(all_sentences):
                    cluster_sentences[f"c{cluster_id}"].append({
                        "sentence": all_sentences[sentence_id],
                        "token": token,
                        "token_idx": token_idx
                    })
    
    return cluster_sentences

def display_aligned_clusters(model_base: str, selected_pair: str, selected_layer: int):
    """Display aligned encoder-decoder cluster pairs with wordclouds and evaluation"""
    st.header(f"Aligned Clusters - Layer {selected_layer}")
    
    # Load the alignments file
    alignments_file = os.path.join(
        model_base,
        selected_pair,
        f"layer{selected_layer}",
        f"Alignments_with_LLM_labels_layer{selected_layer}.json"
    )
    
    # Load cluster alignments metrics file
    metrics_file = os.path.join(
        model_base,
        selected_pair,
        f"layer{selected_layer}",
        "cluster_alignments.json"
    )
    
    if not os.path.exists(alignments_file):
        st.error("No alignment data found for this layer")
        return
        
    with open(alignments_file, 'r') as f:
        alignments = json.load(f)
        
    with open(metrics_file, 'r') as f:
        alignment_metrics = json.load(f)

    # Create dropdown options for cluster pairs first
    cluster_pairs = []
    for src_cluster_id, cluster_data in alignments["alignments"].items():
        encoder_id = cluster_data["encoder_cluster"]["id"]
        for decoder_cluster in cluster_data["aligned_decoder_clusters"]:
            decoder_id = decoder_cluster["id"]
            cluster_pairs.append((encoder_id, decoder_id))
    
    # Dropdown for cluster selection
    selected_pair_idx = st.selectbox(
        "Select cluster pair",
        range(len(cluster_pairs)),
        format_func=lambda x: f"Encoder {cluster_pairs[x][0]} → Decoder {cluster_pairs[x][1]}",
        index=st.session_state.current_cluster_index
    )
    
    # Get the selected encoder and decoder IDs
    selected_encoder_id, selected_decoder_id = cluster_pairs[selected_pair_idx]
    

    
    # The issue might be that the encoder ID in metrics file doesn't include 'c' prefix
    metrics_encoder_id = selected_encoder_id.lstrip('c') if selected_encoder_id.startswith('c') else selected_encoder_id
    
    # Find the corresponding data
    for src_cluster_id, cluster_data in alignments["alignments"].items():
        if cluster_data["encoder_cluster"]["id"] == selected_encoder_id:
            encoder_cluster = cluster_data["encoder_cluster"]
            decoder_cluster = next(
                dc for dc in cluster_data["aligned_decoder_clusters"] 
                if dc["id"] == selected_decoder_id
            )
            break
    
    # Load sentences for both encoder and decoder
    encoder_sentences = load_cluster_sentences(
        os.path.join(model_base, selected_pair),
        selected_layer,
        "encoder"
    )
    
    decoder_sentences = load_cluster_sentences(
        os.path.join(model_base, selected_pair),
        selected_layer,
        "decoder"
    )
    
    # Display clusters side by side with wordclouds
    st.write("### Cluster Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Encoder Cluster")
        # Create and display encoder wordcloud first
        tokens = encoder_cluster.get('unique_tokens', [])
        wc = create_wordcloud(tokens)
        if wc:
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
        # Then display metadata
        st.write(f"**Syntactic Label:** {encoder_cluster.get('syntactic_label', 'N/A')}")
        st.write("**Semantic Tags:**")
        for tag in encoder_cluster.get('semantic_tags', []):
            st.write(f"- {tag}")
        st.write(f"**Description:** {encoder_cluster.get('description', 'N/A')}")
    
    with col2:
        st.write("#### Decoder Cluster")
        # Create and display decoder wordcloud first
        tokens = decoder_cluster.get('unique_tokens', [])
        wc = create_wordcloud(tokens)
        if wc:
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
        # Then display metadata
        st.write(f"**Syntactic Label:** {decoder_cluster.get('syntactic_label', 'N/A')}")
        st.write("**Semantic Tags:**")
        for tag in decoder_cluster.get('semantic_tags', []):
            st.write(f"- {tag}")
        st.write(f"**Description:** {decoder_cluster.get('description', 'N/A')}")
    
    # Display alignment metrics
    st.write("### Alignment Metrics")
    metrics_encoder_id = selected_encoder_id.lstrip('c')  # Remove 'c' prefix if present
    
    # Load cluster alignments metrics file
    metrics_file = os.path.join(
        model_base,
        selected_pair,
        f"layer{selected_layer}",
        "cluster_alignments.json"
    )
    
    if not os.path.exists(metrics_file):
        st.error(f"No metrics file found at {metrics_file}")
        return
        
    with open(metrics_file, 'r') as f:
        alignment_metrics = json.load(f)
    
    if metrics_encoder_id in alignment_metrics:
        metrics = alignment_metrics[metrics_encoder_id]["metrics"]
        column = st.columns(2)  # Create two columns for horizontal layout
            
        with column[0]:
            if "calign_score" in metrics:
                st.metric("Cluster Alignment Score", f"{metrics['calign_score']:.2%}")
        
        with column[1]:  # Add a second column for the next metric
            if "colap_score" in metrics:
                st.metric("Cluster Overlap Score", f"{metrics['colap_score']:.2%}")
    else:
        st.warning(f"No alignment metrics found for cluster {selected_encoder_id}")
        print(f"No metrics found for ID {metrics_encoder_id}")

    # Display evaluation section after cluster details
    st.write("### Alignment Evaluation")
    alignment_accurate = st.radio(
        "Do these clusters align?",
        ["Yes", "No"],
        key=f"align_{selected_pair_idx}"
    )
    
    if alignment_accurate == "Yes":
        alignment_types = st.multiselect(
            "What type of alignment criteria? (Select all that apply)",
            ["Syntactic", "Semantic", "Lexical", "Other"],
            key=f"align_type_{selected_pair_idx}"
        )
    
    notes = st.text_area(
        "Additional notes (optional):" if alignment_accurate == "Yes" and "Other" not in alignment_types else "Please explain why these clusters don't align or specify details for 'Other' alignment type:",
        key=f"align_notes_{selected_pair_idx}"
    )
    
    # Submit button for evaluation
    if st.button("Submit Alignment Evaluation", key=f"submit_{selected_pair_idx}"):
        if alignment_accurate == "No" and not notes.strip():
            st.error("Please provide notes explaining why the clusters don't align.")
        elif alignment_accurate == "Yes" and not alignment_types:
            st.error("Please select at least one alignment type.")
        elif "Other" in alignment_types and not notes.strip():
            st.error("Please provide notes explaining the 'Other' alignment type.")
        else:
            evaluation_data = {
                "encoder_cluster": encoder_cluster['id'],
                "decoder_cluster": decoder_cluster['id'],
                "alignment_accurate": alignment_accurate,
                "alignment_types": alignment_types if alignment_accurate == "Yes" else None,
                "notes": notes
            }
            
            # Extract model from the directory path
            model = model_base.split('/')[0]  # Get just t5 or coderosetta
            language_pair = selected_pair
            
            if save_alignment_evaluation(
                model=model,
                language_pair=language_pair,
                layer_number=selected_layer,
                evaluation_data=evaluation_data
            ):
                st.success("Evaluation submitted successfully!")
                if st.session_state.current_cluster_index < len(cluster_pairs) - 1:
                    st.session_state.current_cluster_index += 1
                    # Add JavaScript to scroll to top before rerun
                    js = '''
                        <script>
                            window.scrollTo(0, 0);
                            var elements = window.parent.document.getElementsByTagName('iframe');
                            for (var i = 0; i < elements.length; i++) {
                                elements[i].contentWindow.scrollTo(0, 0);
                            }
                        </script>
                    '''
                    st.markdown(js, unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.success("All cluster pairs have been evaluated!")
    
    # Display context sentences last
    if encoder_sentences.get(encoder_cluster['id']):
        st.write("### Encoder Context Sentences")
        for sent_info in encoder_sentences[encoder_cluster['id']]:
            tokens = sent_info["sentence"].split()
            html = create_sentence_html(tokens, sent_info)
            st.markdown(html, unsafe_allow_html=True)
            
    if decoder_sentences.get(decoder_cluster['id']):
        st.write("### Decoder Context Sentences")
        for sent_info in decoder_sentences[decoder_cluster['id']]:
            tokens = sent_info["sentence"].split()
            html = create_sentence_html(tokens, sent_info)
            st.markdown(html, unsafe_allow_html=True)

def create_sentence_html(tokens, sent_info):
    """Helper function to create HTML for sentence display"""
    html = """
    <div style='font-family: monospace; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 5px;'>
        <div style='margin-bottom: 5px;'>"""
    
    # If token_idx is provided, highlight that specific token
    # Otherwise, highlight all occurrences of the token
    target_token = sent_info['token'].lower()
    
    for idx, token in enumerate(tokens):
        if ('token_idx' in sent_info and idx == sent_info['token_idx']) or \
           ('token_idx' not in sent_info and token.lower() == target_token):
            html += f"<span style='color: red; font-weight: bold;'>{token}</span> "
        else:
            html += f"{token} "
    
    html += f"""
        </div>
        <div style='color: #666; font-size: 0.9em;'>Token: <code>{sent_info['token']}</code></div>
    </div>
    """
    return html

def save_cluster_evaluation(model: str, language_pair: str, layer_number: int, cluster_id: str, evaluation_data: dict):
    """Save individual cluster evaluation to Supabase"""
    print(f"Saving cluster evaluation for model: {model}, language pair: {language_pair}, layer: {layer_number}, cluster: {cluster_id}")
    try:
        data = {
            "model": model,
            "language_pair": language_pair,
            "layer_number": layer_number,
            "cluster_id": cluster_id,
            "syntactic_accuracy": evaluation_data["syntactic_accuracy"],
            "semantic_accuracy": evaluation_data["semantic_accuracy"],
            "notes": evaluation_data["notes"],
        }
        
        result = supabase.table("cluster_evaluations").upsert(data).execute()
        return True
    except Exception as e:
        st.error(f"Failed to save evaluation: {str(e)}")
        return False

def save_alignment_evaluation(model: str, language_pair: str, layer_number: int, evaluation_data: dict):
    """Save alignment evaluation to Supabase"""
    try:
        data = {
            "model": model,
            "language_pair": language_pair,
            "layer_number": layer_number,
            "encoder_cluster_id": evaluation_data["encoder_cluster"],
            "decoder_cluster_id": evaluation_data["decoder_cluster"],
            "alignment_accurate": evaluation_data["alignment_accurate"] == "Yes",
            "alignment_types": evaluation_data.get("alignment_types"),
            "notes": evaluation_data["notes"],
        }
        
        result = supabase.table("alignment_evaluations").upsert(data).execute()
        return True
    except Exception as e:
        st.error(f"Failed to save evaluation: {str(e)}")
        return False

def display_top_semantic_tags(model_base: str, selected_pair: str):
    """Display top 20 semantic tags for encoder and decoder"""
    st.header("Top 20 Semantic Tags")
    
    # Load top tags files
    encoder_tags_file = os.path.join(model_base, selected_pair, "top_encoder_tags.json")
    decoder_tags_file = os.path.join(model_base, selected_pair, "top_decoder_tags.json")
    
    if not os.path.exists(encoder_tags_file) or not os.path.exists(decoder_tags_file):
        st.error("Top semantic tags files not found. Please run the semantic tag analysis first.")
        return
        
    # Load tag data
    with open(encoder_tags_file, 'r') as f:
        encoder_tags = json.load(f)
    with open(decoder_tags_file, 'r') as f:
        decoder_tags = json.load(f)
        
    # Display tags in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Encoder Top Tags")
        # Create a bar chart for encoder tags
        fig = go.Figure(data=[
            go.Bar(
                x=list(encoder_tags.values()),
                y=list(encoder_tags.keys()),
                orientation='h'
            )
        ])
        fig.update_layout(
            height=600,
            title="Top 20 Encoder Semantic Tags",
            xaxis_title="Frequency",
            yaxis_title="Tag"
        )
        st.plotly_chart(fig)
        
    with col2:
        st.write("### Decoder Top Tags")
        # Create a bar chart for decoder tags
        fig = go.Figure(data=[
            go.Bar(
                x=list(decoder_tags.values()),
                y=list(decoder_tags.keys()),
                orientation='h'
            )
        ])
        fig.update_layout(
            height=600,
            title="Top 20 Decoder Semantic Tags",
            xaxis_title="Frequency",
            yaxis_title="Tag"
        )
        st.plotly_chart(fig)

def get_available_layers(model_base: str, selected_pair: str) -> List[int]:
    """Returns a list of layer numbers that have alignment files or any cluster files."""
    layers = set()  # Using set to avoid duplicates
    pair_dir = os.path.join(model_base, selected_pair)
    
    print(f"Looking for layers in: {pair_dir}")  # Debug print
    
    if not os.path.exists(pair_dir):
        print(f"Directory not found: {pair_dir}")
        return []
        
    # Check each layer directory
    for item in os.listdir(pair_dir):
        if item.startswith('layer'):
            layer_num = int(item.replace('layer', ''))
            layer_dir = os.path.join(pair_dir, item)
            
            # For MLM mixed model or AER mixed model, look for clusters-kmeans-500.txt
            if model_base.startswith(('coderosetta_mlm_mixed', 'coderosetta_aer_mixed')):
                cluster_file = os.path.join(layer_dir, "clusters-kmeans-500.txt")
                if os.path.isfile(cluster_file):
                    try:
                        with open(cluster_file, 'r', encoding='utf-8') as f:
                            next(f)  # Try reading first line
                        layers.add(layer_num)
                        print(f"Found valid mixed cluster file for layer: {layer_num}")
                    except (StopIteration, UnicodeDecodeError) as e:
                        print(f"Error reading mixed cluster file for layer {layer_num}: {str(e)}")
                else:
                    print(f"Cluster file not found at: {cluster_file}")
                continue
            
            # Original logic for other models
            alignment_file = os.path.join(
                layer_dir, 
                f"Alignments_with_LLM_labels_layer{layer_num}.json"
            )
            encoder_cluster_file = os.path.join(layer_dir, "encoder-clusters-kmeans-500.txt")
            decoder_cluster_file = os.path.join(layer_dir, "decoder-clusters-kmeans-500.txt")
            
            # Add layer if it has alignment file
            if os.path.isfile(alignment_file):
                try:
                    with open(alignment_file, 'r', encoding='utf-8') as f:
                        json.load(f)
                    layers.add(layer_num)
                    print(f"Found valid alignment file for layer: {layer_num}")
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error reading alignment file for layer {layer_num}: {str(e)}")
            
            # Check encoder cluster file
            if os.path.isfile(encoder_cluster_file):
                try:
                    with open(encoder_cluster_file, 'r', encoding='utf-8') as f:
                        next(f)  # Try reading first line
                    layers.add(layer_num)
                    print(f"Found valid encoder cluster file for layer: {layer_num}")
                except (StopIteration, UnicodeDecodeError) as e:
                    print(f"Error reading encoder cluster file for layer {layer_num}: {str(e)}")
            
            # Check decoder cluster file
            if os.path.isfile(decoder_cluster_file):
                try:
                    with open(decoder_cluster_file, 'r', encoding='utf-8') as f:
                        next(f)  # Try reading first line
                    layers.add(layer_num)
                    print(f"Found valid decoder cluster file for layer: {layer_num}")
                except (StopIteration, UnicodeDecodeError) as e:
                    print(f"Error reading decoder cluster file for layer {layer_num}: {str(e)}")
    
    return sorted(list(layers))

def validate_selected_layer(layer: int, available_layers: List[int]) -> int:
    """Validates and returns a valid layer number."""
    if not available_layers:
        raise ValueError("No valid layers found")
    
    if layer not in available_layers:
        # Return the first available layer if selected layer is invalid
        return available_layers[0]
    
    return layer

def find_clusters_for_token(model_base: str, selected_pair: str, selected_layer: int, search_token: str):
    """Find all clusters containing the specified token"""
    cluster_file = os.path.join(
        model_base, 
        selected_pair,
        f"layer{selected_layer}",
        "clusters-kmeans-500.txt"
    )
    
    # Dictionary to store clusters containing the token
    token_clusters = {}
    
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|||')
            if len(parts) == 5:  # token|||other|||sent_id|||token_idx|||cluster_id
                token = parts[0].strip()
                cluster_id = parts[4].strip()
                
                if search_token.lower() in token.lower():
                    if f"c{cluster_id}" not in token_clusters:
                        token_clusters[f"c{cluster_id}"] = []
                    token_clusters[f"c{cluster_id}"].append(token)
    
    return token_clusters

def find_clusters_for_token_across_layers(model_base: str, selected_pair: str, available_layers: List[int], search_token: str):
    """Find all clusters containing the specified token across all layers"""
    layer_clusters = {}
    
    for layer in available_layers:
        cluster_file = os.path.join(
            model_base, 
            selected_pair,
            f"layer{layer}",
            "clusters-kmeans-500.txt"
        )
        
        # Dictionary to store clusters and their unique tokens
        token_clusters = {}
        
        with open(cluster_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|||')
                if len(parts) == 5:  # token|||other|||sent_id|||token_idx|||cluster_id
                    token = parts[0].strip()
                    cluster_id = parts[4].strip()
                    
                    if search_token.lower() in token.lower():
                        cluster_key = f"c{cluster_id}"
                        if cluster_key not in token_clusters:
                            token_clusters[cluster_key] = {
                                'matching_tokens': set(),
                                'all_tokens': set()
                            }
                        token_clusters[cluster_key]['matching_tokens'].add(token)
                        token_clusters[cluster_key]['all_tokens'].add(token)
                    elif f"c{cluster_id}" in token_clusters:
                        token_clusters[f"c{cluster_id}"]['all_tokens'].add(token)
        
        if token_clusters:
            layer_clusters[layer] = token_clusters
    
    return layer_clusters

def count_mixed_clusters(model_base: str, selected_pair: str, selected_layer: int) -> int:
    """Count clusters that have mixed language tokens"""
    mixed_count = 0
    
    # Load cluster sentences
    cluster_sentences = load_cluster_sentences(
        os.path.join(model_base, selected_pair),
        selected_layer,
        "mixed"
    )
    
    # For each cluster, check if it has mixed tokens
    for cluster_id, cluster_data in cluster_sentences.items():
        # Extract just the sentences list from the dict structure
        sentences_list = cluster_data["sentences"] if isinstance(cluster_data, dict) else cluster_data
        
        stats = get_language_statistics(sentences_list, os.path.join(model_base, selected_pair))
        if stats and stats["mixed_count"] > 0:
            mixed_count += 1
            
    return mixed_count

def count_language_dominated_clusters(model_base: str, selected_pair: str, selected_layer: int, 
                                    dominance_threshold: float = 0.75,
                                    min_tokens: int = 8) -> dict:
    """Count clusters dominated by each language using proportional thresholds.
    
    Args:
        model_base: Base directory for the model
        selected_pair: Selected language pair
        selected_layer: Layer number to analyze
        dominance_threshold: Proportion of tokens needed to consider a cluster dominated (default 0.75)
        min_tokens: Minimum number of tokens needed for reliable classification (default 8)
    """
    stats = {
        "cpp_dominated": 0,
        "cuda_dominated": 0,
        "mixed": 0,
        "total": 0,
        "small_clusters": 0,  # Clusters with too few tokens
        "detailed_stats": []  # Store detailed statistics for each cluster
    }
    
    cluster_sentences = load_cluster_sentences(
        os.path.join(model_base, selected_pair),
        selected_layer,
        "mixed"
    )
    
    for cluster_id, sentences in cluster_sentences.items():
        sentences_list = sentences["sentences"] if isinstance(sentences, dict) else sentences
        lang_stats = get_language_statistics(sentences_list, os.path.join(model_base, selected_pair))
        
        if lang_stats:
            total_tokens = (lang_stats["cpp_count"] + 
                          lang_stats["cuda_count"] + 
                          lang_stats["mixed_count"])
            
            # Skip clusters with too few tokens
            if total_tokens < min_tokens:
                stats["small_clusters"] += 1
                continue
                
            # Calculate proportions
            cpp_prop = lang_stats["cpp_count"] / total_tokens
            cuda_prop = lang_stats["cuda_count"] / total_tokens
            mixed_prop = lang_stats["mixed_count"] / total_tokens
            
            # Store detailed statistics for this cluster
            cluster_detail = {
                "cluster_id": cluster_id,
                "total_tokens": total_tokens,
                "cpp_proportion": cpp_prop,
                "cuda_proportion": cuda_prop,
                "mixed_proportion": mixed_prop,
                "classification": None
            }
            
            # Determine dominance using thresholds
            if cpp_prop >= dominance_threshold:
                stats["cpp_dominated"] += 1
                cluster_detail["classification"] = "cpp"
            elif cuda_prop >= dominance_threshold:
                stats["cuda_dominated"] += 1
                cluster_detail["classification"] = "cuda"
            else:
                # Consider various mixed scenarios
                if mixed_prop > 0.3:  # Significant mixed tokens
                    stats["mixed"] += 1
                    cluster_detail["classification"] = "truly_mixed"
                elif abs(cpp_prop - cuda_prop) < 0.2:  # Close proportions
                    stats["mixed"] += 1
                    cluster_detail["classification"] = "balanced"
                elif cpp_prop > cuda_prop:
                    stats["cpp_dominated"] += 1
                    cluster_detail["classification"] = "cpp_leaning"
                else:
                    stats["cuda_dominated"] += 1
                    cluster_detail["classification"] = "cuda_leaning"
            
            stats["detailed_stats"].append(cluster_detail)
            stats["total"] += 1
    
    # Add summary statistics
    if stats["total"] > 0:
        stats["summary"] = {
            "cpp_dominated_percent": (stats["cpp_dominated"] / stats["total"]) * 100,
            "cuda_dominated_percent": (stats["cuda_dominated"] / stats["total"]) * 100,
            "mixed_percent": (stats["mixed"] / stats["total"]) * 100,
            "small_clusters_percent": (stats["small_clusters"] / 
                                     (stats["total"] + stats["small_clusters"])) * 100
        }
    
    return stats

def display_language_distribution(model_base, selected_pair, available_layers):
    """Display enhanced language distribution statistics"""
    balance_stats = verify_dataset_balance(os.path.join(model_base, selected_pair))
    if balance_stats:
        st.write("### Dataset Balance")
        cols = st.columns(4)
        with cols[0]:
            st.metric("C++ Sentences", balance_stats["cpp_count"])
        with cols[1]:
            st.metric("CUDA Sentences", balance_stats["cuda_count"])
        with cols[2]:
            st.metric("Unknown", balance_stats["unknown_count"])
        with cols[3]:
            st.metric("Total", balance_stats["total_count"])
        st.markdown("---")
    
    st.write("### Layer-wise Language Distribution")
    
    # Add controls for analysis parameters
    col1, col2 = st.columns(2)
    with col1:
        dominance_threshold = st.slider(
            "Dominance Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.75,
            step=0.05,
            help="Proportion of tokens needed to consider a cluster dominated by a language"
        )
    with col2:
        min_tokens = st.slider(
            "Minimum Tokens",
            min_value=3,
            max_value=20,
            value=8,
            help="Minimum number of tokens needed for reliable classification"
        )
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary View", "Detailed View", "Layerwise Graph"])
    
    # Store data for plotting
    layer_data = {
        'layers': [],
        'cpp_dominated': [],
        'cuda_dominated': [],
        'mixed': [],
        'small_clusters': []
    }
    
    with tab1:
        for layer in available_layers:
            stats = count_language_dominated_clusters(
                model_base, 
                selected_pair, 
                layer,
                dominance_threshold,
                min_tokens
            )
            
            st.write(f"#### Layer {layer}")
            cols = st.columns(5)
            
            with cols[0]:
                st.metric("C++ Dominated", 
                         f"{stats['cpp_dominated']} ({stats['summary']['cpp_dominated_percent']:.1f}%)")
            with cols[1]:
                st.metric("CUDA Dominated", 
                         f"{stats['cuda_dominated']} ({stats['summary']['cuda_dominated_percent']:.1f}%)")
            with cols[2]:
                st.metric("Mixed", 
                         f"{stats['mixed']} ({stats['summary']['mixed_percent']:.1f}%)")
            with cols[3]:
                st.metric("Total Clusters", stats['total'])
            with cols[4]:
                st.metric("Small Clusters", 
                         f"{stats['small_clusters']} ({stats['summary']['small_clusters_percent']:.1f}%)")
    
    with tab2:
        for layer in available_layers:
            stats = count_language_dominated_clusters(
                model_base, 
                selected_pair, 
                layer,
                dominance_threshold,
                min_tokens
            )
            
            st.write(f"#### Layer {layer}")
            
            # Convert detailed stats to DataFrame for better visualization
            if stats['detailed_stats']:
                import pandas as pd
                df = pd.DataFrame(stats['detailed_stats'])
                df = df.round(3)  # Round proportions to 3 decimal places
                
                # Color-code the classification column
                def color_classification(val):
                    colors = {
                        'cpp': 'background-color: #90EE90',
                        'cuda': 'background-color: #87CEEB',
                        'truly_mixed': 'background-color: #DDA0DD',
                        'balanced': 'background-color: #F0E68C',
                        'cpp_leaning': 'background-color: #98FB98',
                        'cuda_leaning': 'background-color: #ADD8E6'
                    }
                    return colors.get(val, '')
                
                styled_df = df.style.apply(lambda x: [color_classification(v) for v in x], 
                                         subset=['classification'])
                
                st.dataframe(styled_df)
    
    with tab3:
        st.write(f"### Layerwise Distribution (Threshold: {dominance_threshold:.2f}, Min Tokens: {min_tokens})")
        
        # Collect data for all layers
        for layer in available_layers:
            stats = count_language_dominated_clusters(
                model_base, 
                selected_pair, 
                layer,
                dominance_threshold,
                min_tokens
            )
            
            layer_data['layers'].append(layer)
            layer_data['cpp_dominated'].append(stats['summary']['cpp_dominated_percent'])
            layer_data['cuda_dominated'].append(stats['summary']['cuda_dominated_percent'])
            layer_data['mixed'].append(stats['summary']['mixed_percent'])
            layer_data['small_clusters'].append(stats['summary']['small_clusters_percent'])
        
        # Create the figure
        fig = go.Figure()
        
        # Add traces for each category
        fig.add_trace(go.Scatter(
            x=layer_data['layers'],
            y=layer_data['cpp_dominated'],
            name='C++ Dominated',
            mode='lines+markers',
            line=dict(color='#90EE90', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=layer_data['layers'],
            y=layer_data['cuda_dominated'],
            name='CUDA Dominated',
            mode='lines+markers',
            line=dict(color='#87CEEB', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=layer_data['layers'],
            y=layer_data['mixed'],
            name='Mixed',
            mode='lines+markers',
            line=dict(color='#DDA0DD', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=layer_data['layers'],
            y=layer_data['small_clusters'],
            name='Small Clusters',
            mode='lines+markers',
            line=dict(color='#808080', width=2),
            marker=dict(size=8)
        ))
        
        # Update layout with detailed title including model name
        fig.update_layout(
            title=dict(
                text=f'Language Distribution Across Layers - {model_base}<br><sup>Dominance Threshold: {dominance_threshold:.2f}, Minimum Tokens: {min_tokens}</sup>',
                font=dict(weight='bold', size=20),
                y=0.95,  # Adjust title position to accommodate subtitle
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis_title=dict(
                text='Layer',
                font=dict(weight='bold', size=14)
            ),
            yaxis_title=dict(
                text='Percentage (%)',
                font=dict(weight='bold', size=14)
            ),
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(weight='bold')
            )
        )
        
        # Add gridlines
        fig.update_xaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
        fig.update_yaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add download buttons
        col1, col2 = st.columns(2)
        
        # Download plot as HTML
        with col1:
            buffer = io.StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Plot as HTML",
                data=html_bytes,
                file_name=f"layer_distribution_t{dominance_threshold}_m{min_tokens}.html",
                mime="text/html"
            )
        
        # Download data as CSV
        with col2:
            df = pd.DataFrame({
                'Layer': layer_data['layers'],
                'CPP_Dominated_%': layer_data['cpp_dominated'],
                'CUDA_Dominated_%': layer_data['cuda_dominated'],
                'Mixed_%': layer_data['mixed'],
                'Small_Clusters_%': layer_data['small_clusters']
            })
            
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"layer_distribution_t{dominance_threshold}_m{min_tokens}.csv",
                mime="text/csv"
            )

        # Add new plot for nuanced classifications
        st.write("### Nuanced Classification Distribution")
        
        # Initialize data structure for nuanced classifications - removed pure categories
        nuanced_data = {
            'layers': [],
            'cpp_leaning': [],
            'cuda_leaning': [],
            'truly_mixed': []
        }
        
        # Collect nuanced classification data
        for layer in available_layers:
            stats = count_language_dominated_clusters(
                model_base, 
                selected_pair, 
                layer,
                dominance_threshold,
                min_tokens
            )
            
            # Count occurrences of each classification - removed pure and balanced categories
            classifications = {
                'cpp_leaning': 0, 
                'cuda_leaning': 0, 
                'truly_mixed': 0
            }
            
            total_classified = 0
            for cluster_stat in stats['detailed_stats']:
                if cluster_stat['classification'] in classifications:
                    classifications[cluster_stat['classification']] += 1
                    total_classified += 1
            
            # Convert to percentages
            if total_classified > 0:
                nuanced_data['layers'].append(layer)
                for key in classifications:
                    nuanced_data[key].append((classifications[key] / total_classified) * 100)
        
        # Create nuanced classification figure
        fig_nuanced = go.Figure()
        
        # Add traces for each classification with distinct colors - removed pure categories
        colors = {
            'cpp_leaning': '#98FB98',  # Pale green
            'cuda_leaning': '#ADD8E6',  # Light blue
            'truly_mixed': '#DDA0DD',  # Plum
        }
        
        for classification, color in colors.items():
            fig_nuanced.add_trace(go.Scatter(
                x=nuanced_data['layers'],
                y=nuanced_data[classification],
                name=classification.replace('_', ' ').title(),
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ))
        
        # Update layout for nuanced plot with detailed title including model name
        fig_nuanced.update_layout(
            title=dict(
                text=f'Intermediate Classification Distribution Across Layers - {model_base}<br><sup>Dominance Threshold: {dominance_threshold:.2f}, Minimum Tokens: {min_tokens}</sup>',
                font=dict(weight='bold', size=20),
                y=0.95,  # Adjust title position to accommodate subtitle
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis_title=dict(
                text='Layer',
                font=dict(weight='bold', size=14)
            ),
            yaxis_title=dict(
                text='Percentage (%)',
                font=dict(weight='bold', size=14)
            ),
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(weight='bold')
            )
        )
        
        # Add gridlines
        fig_nuanced.update_xaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
        fig_nuanced.update_yaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
        
        # Display the nuanced plot
        st.plotly_chart(fig_nuanced, use_container_width=True)
        
        # Add download buttons for nuanced plot
        col3, col4 = st.columns(2)
        
        # Download nuanced plot as HTML
        with col3:
            buffer = io.StringIO()
            fig_nuanced.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Nuanced Plot as HTML",
                data=html_bytes,
                file_name=f"intermediate_distribution_t{dominance_threshold}_m{min_tokens}.html",
                mime="text/html"
            )
        
        # Download nuanced data as CSV
        with col4:
            df_nuanced = pd.DataFrame({
                'Layer': nuanced_data['layers'],
                'CPP_Leaning_%': nuanced_data['cpp_leaning'],
                'CUDA_Leaning_%': nuanced_data['cuda_leaning'],
                'Truly_Mixed_%': nuanced_data['truly_mixed']
            })
            
            csv_nuanced = df_nuanced.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Nuanced Data as CSV",
                data=csv_nuanced,
                file_name=f"intermediate_distribution_t{dominance_threshold}_m{min_tokens}.csv",
                mime="text/csv"
            )

def handle_token_search(model_name, model_base, selected_pair, available_layers):
    """Handle token search functionality"""
    if 'search_text' not in st.session_state:
        st.session_state.search_text = ""
    if 'search_key' not in st.session_state:
        st.session_state.search_key = 0
    if 'selected_token' not in st.session_state:
        st.session_state.selected_token = None
        
    st.sidebar.write("### Token Search")
    search_token = st.sidebar.text_input(
        "Search for token:", 
        value=st.session_state.search_text,
        key=f"token_search_input_{st.session_state.search_key}"
    )
    
    if st.sidebar.button("Clear", key="clear_search"):
        st.session_state.search_text = ""
        st.session_state.search_key += 1
        st.session_state.selected_token = None
        if 'layer_results' in st.session_state:
            st.session_state.layer_results = {}
        st.rerun()
    
    # Check if search text has changed
    if search_token != st.session_state.search_text:
        st.session_state.search_text = search_token
        st.session_state.selected_token = None
        if 'layer_results' in st.session_state:
            st.session_state.layer_results = {}

    if search_token and search_token.strip():
        display_search_results(model_name, model_base, selected_pair, available_layers, search_token)
        return True
    return False

def display_search_results(model_name, model_base, selected_pair, available_layers, search_token):
    """Display search results for a token across all layers"""
    st.write(f"### Search Results for '{search_token}'")
    
    # Find all matching tokens across all layers first
    all_matching_tokens = set()
    for layer in available_layers:
        layer_results = find_clusters_for_token_across_layers(
            model_base, 
            selected_pair, 
            [layer], 
            search_token
        ).get(layer, {})
        
        for cluster_data in layer_results.values():
            all_matching_tokens.update(cluster_data['matching_tokens'])
    
    # Sort matching tokens for consistent display
    matching_tokens_list = sorted(all_matching_tokens)
    
    if not matching_tokens_list:
        st.warning("No matching tokens found")
        return
        
    # Display token selection
    selected_token = st.selectbox(
        "Select a specific token to view its clusters:",
        matching_tokens_list,
        key="token_selector"
    )
    
    if selected_token != st.session_state.selected_token:
        st.session_state.selected_token = selected_token
        if 'layer_results' in st.session_state:
            st.session_state.layer_results = {}
    
    # Only proceed if a token is selected
    if selected_token:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Evolution Analysis", "Cluster View"])
        
        with tab1:
            # Display evolution analysis for the selected token
            with st.spinner(f"Analyzing evolution of '{selected_token}' across layers..."):
                evolution_data = analyze_keyword_evolution(
                    model_base,
                    selected_pair,
                    available_layers,
                    selected_token
                )
                
                if any(evolution_data['cluster_counts']):
                    display_keyword_evolution(evolution_data, selected_token, context="search")
                else:
                    st.warning(f"No occurrences of '{selected_token}' found in any layer")
        
        with tab2:
            # Initialize layer results in session state if not present
            if 'layer_results' not in st.session_state:
                st.session_state.layer_results = {}
                
            # Create tabs for each layer
            tab_labels = [f"Layer {layer}" for layer in available_layers]
            active_tab = st.tabs(tab_labels)
            
            # Display results for each layer
            for idx, layer in enumerate(available_layers):
                with active_tab[idx]:
                    # Only compute results when tab is clicked
                    if layer not in st.session_state.layer_results:
                        with st.spinner(f"Computing results for Layer {layer}..."):
                            layer_results = find_clusters_for_token_across_layers(
                                model_base, 
                                selected_pair, 
                                [layer], 
                                selected_token
                            ).get(layer, {})
                            
                            # Filter results to only include clusters containing the exact selected token
                            filtered_results = {}
                            for cluster_id, tokens in layer_results.items():
                                if selected_token in tokens['matching_tokens']:
                                    filtered_results[cluster_id] = tokens
                            
                            st.session_state.layer_results[layer] = filtered_results
                    
                    layer_results = st.session_state.layer_results[layer]
                    
                    if not layer_results:
                        st.write(f"No matches found in layer {layer}")
                        continue
                    
                    # Display results for each cluster
                    for cluster_id, tokens in layer_results.items():
                        cluster_heading = f"Cluster {cluster_id}"
                        
                        with st.expander(cluster_heading):
                            # Create word cloud for all tokens in cluster
                            all_tokens = list(tokens['all_tokens'])
                            if all_tokens:
                                st.write("#### Word Cloud of Cluster Tokens")
                                wc = create_wordcloud(all_tokens)
                                if wc:
                                    fig = plt.figure(figsize=(10, 5))
                                    plt.imshow(wc, interpolation='bilinear')
                                    plt.axis('off')
                                    st.pyplot(fig)
                                    plt.close(fig)
                            
                            st.write("#### All Tokens in Cluster")
                            st.write(", ".join(sorted(tokens['all_tokens'])))
                            
                            # Load and display sentences for this cluster
                            cluster_sentences = load_cluster_sentences(
                                os.path.join(model_base, selected_pair),
                                layer,
                                "mixed"
                            )
                            
                            if cluster_sentences and cluster_id in cluster_sentences:
                                st.write("#### Context Sentences")
                                context_sentences = cluster_sentences[cluster_id]
                                if isinstance(context_sentences, dict):
                                    context_sentences = context_sentences.get('sentences', [])
                                
                                # Separate sentences into matching and non-matching
                                matching_sentences = []
                                other_sentences = []
                                
                                for sent_info in context_sentences:
                                    if selected_token.lower() in sent_info['sentence'].lower():
                                        matching_sentences.append(sent_info)
                                    else:
                                        other_sentences.append(sent_info)
                                
                                # Display matching sentences first
                                if matching_sentences:
                                    st.write("**Sentences with Selected Token:**")
                                    for sent_info in matching_sentences:
                                        html = create_sentence_html(sent_info['sentence'].split(), sent_info)
                                        st.markdown(html, unsafe_allow_html=True)
                                
                                # Display other sentences with a toggle
                                if other_sentences:
                                    show_others = st.checkbox(
                                        f"Show Other Context Sentences ({len(other_sentences)} sentences)", 
                                        key=f"show_others_{layer}_{cluster_id}"
                                    )
                                    if show_others:
                                        st.write("**Other Context Sentences:**")
                                        for sent_info in other_sentences:
                                            html = create_sentence_html(sent_info['sentence'].split(), sent_info)
                                            st.markdown(html, unsafe_allow_html=True)

def verify_dataset_balance(model_dir: str) -> dict:
    """Verify the balance of C++ and CUDA sentences in the shuffled dataset"""
    shuffled_file = os.path.join(model_dir, "shuffled_dataset.txt")
    cpp_file = os.path.join(model_dir, "input.in")
    cuda_file = os.path.join(model_dir, "label.out")
    
    if not all(os.path.exists(f) for f in [shuffled_file, cpp_file, cuda_file]):
        return None
        
    # Load all sentences
    with open(cpp_file, 'r', encoding='utf-8') as f:
        cpp_sentences = set(line.strip() for line in f)
    with open(cuda_file, 'r', encoding='utf-8') as f:
        cuda_sentences = set(line.strip() for line in f)
    with open(shuffled_file, 'r', encoding='utf-8') as f:
        shuffled_sentences = [line.strip() for line in f]
    
    # Count sentences by type
    stats = {
        "cpp_count": 0,
        "cuda_count": 0,
        "unknown_count": 0,
        "total_count": len(shuffled_sentences)
    }
    
    for sentence in shuffled_sentences:
        if sentence in cpp_sentences:
            stats["cpp_count"] += 1
        elif sentence in cuda_sentences:
            stats["cuda_count"] += 1
        else:
            stats["unknown_count"] += 1
    
    return stats

def display_standard_clusters(model_name, model_base, selected_pair, selected_layer, component):
    """Display clusters for standard (non-mixed) models"""
    # Load cluster data
    cluster_file = os.path.join(
        model_base,
        selected_pair,
        f"layer{selected_layer}",
        f"{component}-clusters-kmeans-500.txt"
    )
    
    if not os.path.exists(cluster_file):
        st.error(f"No cluster data found for {component} at layer {selected_layer}")
        return
        
    # Load sentences
    sentences = load_cluster_sentences(
        os.path.join(model_base, selected_pair),
        selected_layer,
        component
    )
    
    if not sentences:
        st.error(f"No sentence data found for {component}")
        return
        
    # Get list of clusters
    clusters = sorted(sentences.keys(), key=lambda x: int(x[1:]))  # Sort by cluster number
    
    # Cluster selection
    selected_cluster = st.sidebar.selectbox(
        "Select Cluster",
        clusters,
        format_func=lambda x: f"Cluster {x[1:]}",  # Remove 'c' prefix for display
        index=min(st.session_state.current_cluster_index, len(clusters)-1)
    )
    
    if selected_cluster:
        # Display cluster information
        st.write(f"### {component.title()} Cluster {selected_cluster[1:]}")
        
        # Create word cloud from sentences in this cluster
        cluster_sentences = sentences[selected_cluster]
        tokens = set()
        for sent_info in cluster_sentences:
            tokens.add(sent_info["token"])
            
        if tokens:
            st.write("#### Word Cloud")
            wc = create_wordcloud(list(tokens))
            if wc:
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)
        
        # Display context sentences
        st.write("#### Context Sentences")
        for sent_info in cluster_sentences:
            html = create_sentence_html(sent_info["sentence"].split(), sent_info)
            st.markdown(html, unsafe_allow_html=True)

def display_mixed_clusters(model_name, model_base, selected_pair, selected_layer):
    """Display clusters for mixed models"""
    # Load cluster data
    cluster_file = os.path.join(
        model_base,
        selected_pair,
        f"layer{selected_layer}",
        "clusters-kmeans-500.txt"
    )
    
    if not os.path.exists(cluster_file):
        st.error(f"No cluster data found at layer {selected_layer}")
        return
        
    # Load sentences
    sentences = load_cluster_sentences(
        os.path.join(model_base, selected_pair),
        selected_layer,
        "mixed"
    )
    
    if not sentences:
        st.error("No sentence data found")
        return
        
    # Get list of clusters
    clusters = sorted(sentences.keys(), key=lambda x: int(x[1:]))  # Sort by cluster number
    
    # Cluster selection
    selected_cluster = st.sidebar.selectbox(
        "Select Cluster",
        clusters,
        format_func=lambda x: f"Cluster {x[1:]}",  # Remove 'c' prefix for display
        index=min(st.session_state.current_cluster_index, len(clusters)-1)
    )
    
    if selected_cluster:
        # Display cluster information
        st.write(f"### Cluster {selected_cluster[1:]}")
        
        # Get cluster sentences and tokens
        cluster_data = sentences[selected_cluster]
        if isinstance(cluster_data, dict):
            cluster_sentences = cluster_data.get('sentences', [])
            unique_tokens = cluster_data.get('unique_tokens', [])
        else:
            cluster_sentences = cluster_data
            unique_tokens = list({sent_info["token"] for sent_info in cluster_data})
            
        # Create word cloud
        if unique_tokens:
            st.write("#### Word Cloud")
            wc = create_wordcloud(unique_tokens)
            if wc:
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                
        # Display language statistics
        stats = get_language_statistics(cluster_sentences, os.path.join(model_base, selected_pair))
        if stats:
            st.write("#### Language Distribution")
            total = stats["total_tokens"]
            
            # Create metrics for token counts
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("C++ Tokens", f"{stats['cpp_count']} ({(stats['cpp_count']/total)*100:.1f}%)")
                st.metric("Unique C++ Tokens", stats['unique_cpp_tokens'])
            with col2:
                st.metric("CUDA Tokens", f"{stats['cuda_count']} ({(stats['cuda_count']/total)*100:.1f}%)")
                st.metric("Unique CUDA Tokens", stats['unique_cuda_tokens'])
            with col3:
                st.metric("Mixed Tokens", f"{stats['mixed_count']} ({(stats['mixed_count']/total)*100:.1f}%)")
                st.metric("Unique Mixed Tokens", stats['unique_mixed_tokens'])
            with col4:
                st.metric("Unknown", f"{stats['unknown_count']} ({(stats['unknown_count']/total)*100:.1f}%)")
                st.metric("Unique Unknown Tokens", stats['unique_unknown_tokens'])
        
        # Display context sentences
        st.write("#### Context Sentences")
        tab1, tab2, tab3, tab4 = st.tabs(["C++", "CUDA", "Mixed", "Unknown"])
        
        with tab1:
            if stats and stats["cpp_sentences"]:
                for token, sentence in stats["cpp_sentences"]:
                    html = create_sentence_html(sentence.split(), {"sentence": sentence, "token": token})
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.write("No C++ sentences found")
                
        with tab2:
            if stats and stats["cuda_sentences"]:
                for token, sentence in stats["cuda_sentences"]:
                    html = create_sentence_html(sentence.split(), {"sentence": sentence, "token": token})
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.write("No CUDA sentences found")
                
        with tab3:
            if stats and stats["mixed_sentences"]:
                for token, sentence in stats["mixed_sentences"]:
                    html = create_sentence_html(sentence.split(), {"sentence": sentence, "token": token})
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.write("No mixed sentences found")
                
        with tab4:
            if stats and stats["unknown_sentences"]:
                for token, sentence in stats["unknown_sentences"]:
                    html = create_sentence_html(sentence.split(), {"sentence": sentence, "token": token})
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.write("No unknown sentences found")

def analyze_keyword_evolution(model_base: str, selected_pair: str, available_layers: List[int], keyword: str):
    """Analyze and visualize how a specific keyword evolves across layers"""
    
    # Data structure to store analysis results
    evolution_data = {
        'layers': [],
        'cluster_counts': [],  # Number of clusters containing the keyword
        'token_counts': [],    # Total occurrences of the keyword
        'cluster_details': {}  # Detailed information about each cluster containing the keyword
    }
    
    # Analyze each layer
    for layer in available_layers:
        # Load cluster data for this layer
        layer_results = find_clusters_for_token_across_layers(
            model_base,
            selected_pair,
            [layer],
            keyword
        ).get(layer, {})
        
        # Count clusters and token occurrences
        clusters_with_keyword = 0
        total_token_occurrences = 0
        cluster_info = {}
        
        for cluster_id, tokens in layer_results.items():
            if keyword in tokens['matching_tokens']:
                clusters_with_keyword += 1
                token_count = sum(1 for t in tokens['all_tokens'] if t == keyword)
                total_token_occurrences += token_count
                
                # Store detailed information about this cluster
                cluster_info[cluster_id] = {
                    'token_count': token_count,
                    'cluster_size': len(tokens['all_tokens']),
                    'token_percentage': token_count / len(tokens['all_tokens']) * 100
                }
        
        # Store data for this layer
        evolution_data['layers'].append(layer)
        evolution_data['cluster_counts'].append(clusters_with_keyword)
        evolution_data['token_counts'].append(total_token_occurrences)
        evolution_data['cluster_details'][layer] = cluster_info
    
    return evolution_data

def display_keyword_evolution(evolution_data: dict, keyword: str, context: str = "search"):
    """
    Display visualizations and analysis of keyword evolution
    Args:
        evolution_data: Dictionary containing evolution analysis data
        keyword: The keyword being analyzed
        context: String identifier for the context ("search" or "predefined")
    """
    # Generate a unique suffix for this display instance
    unique_suffix = str(int(time.time() * 1000))
    
    st.write(f"### Evolution Analysis for '{keyword}'")
    
    # Create main evolution graph
    fig = go.Figure()
    
    # Add cluster count trace
    fig.add_trace(go.Scatter(
        x=evolution_data['layers'],
        y=evolution_data['cluster_counts'],
        name='Clusters with Keyword',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    # Add token count trace
    fig.add_trace(go.Scatter(
        x=evolution_data['layers'],
        y=evolution_data['token_counts'],
        name='Total Token Occurrences',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Update layout with two y-axes using correct property names
    fig.update_layout(
        title=f"Evolution of '{keyword}' Across Layers",
        xaxis=dict(title='Layer'),
        yaxis=dict(
            title=dict(
                text='Number of Clusters',
                font=dict(color='#1f77b4')
            ),
            tickfont=dict(color='#1f77b4')
        ),
        yaxis2=dict(
            title=dict(
                text='Total Token Occurrences',
                font=dict(color='#ff7f0e')
            ),
            tickfont=dict(color='#ff7f0e'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed statistics
    st.write("### Detailed Statistics")
    
    # Create a DataFrame for the statistics
    stats_data = {
        'Layer': evolution_data['layers'],
        'Clusters with Keyword': evolution_data['cluster_counts'],
        'Total Occurrences': evolution_data['token_counts'],
        'Avg Occurrences per Cluster': [
            round(t/c, 2) if c > 0 else 0 
            for t, c in zip(evolution_data['token_counts'], evolution_data['cluster_counts'])
        ]
    }
    
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats)
    
    # Create heatmap of cluster distributions
    st.write("### Cluster Distribution Heatmap")
    
    # Prepare data for heatmap
    heatmap_data = []
    max_clusters = max(len(details) for details in evolution_data['cluster_details'].values())
    
    for layer in evolution_data['layers']:
        layer_data = evolution_data['cluster_details'][layer]
        row = []
        for cluster_id in sorted(layer_data.keys(), key=lambda x: int(x[1:])):
            row.append(layer_data[cluster_id]['token_percentage'])
        # Pad with zeros if needed
        row.extend([0] * (max_clusters - len(row)))
        heatmap_data.append(row)
    
    # Create heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        y=evolution_data['layers'],
        x=[f'Cluster {i+1}' for i in range(max_clusters)],
        colorscale='Viridis',
        colorbar=dict(title='Token %')
    ))
    
    fig_heatmap.update_layout(
        title=f"Distribution of '{keyword}' Across Clusters and Layers",
        xaxis_title="Clusters",
        yaxis_title="Layer",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Add download buttons for the data with unique timestamp-based keys
    col1, col2 = st.columns(2)
    
    with col1:
        # Download statistics as CSV with unique key
        csv = df_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Statistics as CSV",
            data=csv,
            file_name=f"keyword_evolution_{keyword}.csv",
            mime="text/csv",
            key=f"{context}_csv_download_{keyword}_{unique_suffix}"  # Added timestamp
        )
    
    with col2:
        # Download full analysis as JSON with unique key
        json_str = json.dumps(evolution_data, indent=2)
        st.download_button(
            label="Download Full Analysis as JSON",
            data=json_str,
            file_name=f"keyword_evolution_{keyword}.json",
            mime="application/json",
            key=f"{context}_json_download_{keyword}_{unique_suffix}"  # Added timestamp
        )

def add_keyword_evolution_section(model_name, model_base, selected_pair, available_layers):
    """Add a section for keyword evolution analysis"""
    st.write("## Keyword Evolution Analysis")
    
    keyword = st.text_input("Enter keyword to analyze:", key="keyword_evolution_input")
    
    if keyword:
        with st.spinner(f"Analyzing evolution of '{keyword}' across layers..."):
            evolution_data = analyze_keyword_evolution(
                model_base,
                selected_pair,
                available_layers,
                keyword
            )
            
            if any(evolution_data['cluster_counts']):
                display_keyword_evolution(evolution_data, keyword)
            else:
                st.warning(f"No occurrences of '{keyword}' found in any layer")

def analyze_predefined_keywords(model_base: str, selected_pair: str, available_layers: List[int]):
    """Analyze evolution of predefined CUDA and C++ keywords"""
    
    # Define the keywords
    cuda_top8 = [
        "__global__",  # Defines a function that runs on the GPU and is called from the CPU
        "__device__",  # Defines a function that runs on the GPU and is called from the GPU
        "__host__",   # Specifies a function that runs on the CPU
        "__shared__", # Declares shared memory accessible by all threads in a block
        "__constant__", # Declares constant memory on the GPU
        "threadIdx",  # Built-in variable providing thread index within a block
        "blockIdx",   # Built-in variable providing block index within a grid
        "gridDim"     # Built-in variable providing the number of blocks in a grid
    ]

    cpp_top8 = [
        "class",     # Defines a class for object-oriented programming
        "template",  # Enables generic programming
        "constexpr", # Compile-time constant evaluation
        "virtual",   # Supports polymorphism in classes
        "override",  # Ensures a function properly overrides a base class method
        "new",       # Allocates memory dynamically
        "delete",    # Deallocates dynamically allocated memory
        "namespace"  # Helps organize code and prevent naming conflicts
    ]
    
    # Analyze evolution for each keyword
    cuda_evolution = {}
    cpp_evolution = {}
    
    with st.spinner("Analyzing CUDA keywords..."):
        for keyword in cuda_top8:
            cuda_evolution[keyword] = analyze_keyword_evolution(
                model_base,
                selected_pair,
                available_layers,
                keyword
            )
    
    with st.spinner("Analyzing C++ keywords..."):
        for keyword in cpp_top8:
            cpp_evolution[keyword] = analyze_keyword_evolution(
                model_base,
                selected_pair,
                available_layers,
                keyword
            )
    
    return cuda_evolution, cpp_evolution

def display_predefined_keywords_analysis(cuda_evolution: dict, cpp_evolution: dict, available_layers: List[int]):
    """Display analysis of predefined keywords"""
    
    tab1, tab2, tab3 = st.tabs(["Combined View", "CUDA Keywords", "C++ Keywords"])
    
    with tab1:
        st.write("### Combined Keywords Evolution")
        
        # Create combined graph
        fig = go.Figure()
        
        # Add CUDA keywords
        for keyword, data in cuda_evolution.items():
            fig.add_trace(go.Scatter(
                x=data['layers'],
                y=data['cluster_counts'],
                name=f"CUDA: {keyword}",
                mode='lines+markers',
                line=dict(dash='solid'),
                marker=dict(size=8)
            ))
            
        # Add C++ keywords
        for keyword, data in cpp_evolution.items():
            fig.add_trace(go.Scatter(
                x=data['layers'],
                y=data['cluster_counts'],
                name=f"C++: {keyword}",
                mode='lines+markers',
                line=dict(dash='dot'),
                marker=dict(size=8)
            ))
            
        fig.update_layout(
            title="Evolution of Keywords Across Layers",
            xaxis_title="Layer",
            yaxis_title="Number of Clusters",
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=-0.1,
                xanchor="left",
                x=0,
                orientation="h"
            ),
            margin=dict(b=150)  # Add bottom margin for legend
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.write("### CUDA Keywords Evolution")
        
        # Create heatmap for CUDA keywords
        heatmap_data = []
        for keyword in cuda_evolution:
            row = []
            for layer in available_layers:
                count = cuda_evolution[keyword]['cluster_counts'][
                    cuda_evolution[keyword]['layers'].index(layer)
                ]
                row.append(count)
            heatmap_data.append(row)
            
        fig_cuda = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=available_layers,
            y=list(cuda_evolution.keys()),
            colorscale='Viridis',
            colorbar=dict(title='Clusters')
        ))
        
        fig_cuda.update_layout(
            title="CUDA Keywords Distribution Across Layers",
            xaxis_title="Layer",
            yaxis_title="Keyword",
            height=600
        )
        
        st.plotly_chart(fig_cuda, use_container_width=True)
        
        # Individual CUDA keyword graphs
        for keyword, data in cuda_evolution.items():
            with st.expander(f"Detailed View: {keyword}"):
                display_keyword_evolution(data, keyword)
        
    with tab3:
        st.write("### C++ Keywords Evolution")
        
        # Create heatmap for C++ keywords
        heatmap_data = []
        for keyword in cpp_evolution:
            row = []
            for layer in available_layers:
                count = cpp_evolution[keyword]['cluster_counts'][
                    cpp_evolution[keyword]['layers'].index(layer)
                ]
                row.append(count)
            heatmap_data.append(row)
            
        fig_cpp = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=available_layers,
            y=list(cpp_evolution.keys()),
            colorscale='Viridis',
            colorbar=dict(title='Clusters')
        ))
        
        fig_cpp.update_layout(
            title="C++ Keywords Distribution Across Layers",
            xaxis_title="Layer",
            yaxis_title="Keyword",
            height=600
        )
        
        st.plotly_chart(fig_cpp, use_container_width=True)
        
        # Individual C++ keyword graphs
        for keyword, data in cpp_evolution.items():
            with st.expander(f"Detailed View: {keyword}"):
                display_keyword_evolution(data, keyword)

def add_predefined_keywords_tab(model_name, model_base, selected_pair, available_layers):
    """Add predefined keywords analysis tab"""
    st.write("## Predefined Keywords Analysis")
    
    # Add a refresh button
    if st.button("Refresh Analysis"):
        if 'cuda_evolution' in st.session_state:
            del st.session_state.cuda_evolution
        if 'cpp_evolution' in st.session_state:
            del st.session_state.cpp_evolution
    
    # Use session state to cache results
    if 'cuda_evolution' not in st.session_state or 'cpp_evolution' not in st.session_state:
        cuda_evolution, cpp_evolution = analyze_predefined_keywords(
            model_base,
            selected_pair,
            available_layers
        )
        st.session_state.cuda_evolution = cuda_evolution
        st.session_state.cpp_evolution = cpp_evolution
    
    display_predefined_keywords_analysis(
        st.session_state.cuda_evolution,
        st.session_state.cpp_evolution,
        available_layers
    )

def main():
    st.set_page_config(layout="wide", page_title="Code Concept Explorer")
    
    st.title("Code Concept Cluster Explorer")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["t5", "coderosetta", "coderosetta_aer", "coderosetta_mlm", "coderosetta_mlm_mixed", "coderosetta_aer_mixed"],
        key="model_select"
    )
    model_base = os.path.join(model_name)
    
    # Get available language pairs
    lang_pairs = [d for d in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, d))]
    if not lang_pairs:
        st.error("No language pairs found in the specified directory")
        return
        
    selected_pair = st.sidebar.selectbox("Language Pair", lang_pairs, key="pair_select")
    
    # Get available layers
    available_layers = get_available_layers(model_base, selected_pair)
    
    if not available_layers:
        st.error("No layers found with valid data")
        return
    
    # Get and validate selected layer
    selected_layer = st.sidebar.selectbox(
        "Layer",
        available_layers,
        format_func=lambda x: f"Layer {x}",
        key="layer_select"
    )
    
    if selected_layer is None and available_layers:
        selected_layer = available_layers[0]

    # Initialize session state for cluster index if not exists
    if 'current_cluster_index' not in st.session_state:
        st.session_state.current_cluster_index = 0

    # Split logic based on model type
    is_mixed_model = model_name in ["coderosetta_mlm_mixed", "coderosetta_aer_mixed"]
    
    if is_mixed_model:
        handle_mixed_model_view(model_name, model_base, selected_pair, selected_layer, available_layers)
    else:
        handle_standard_model_view(model_name, model_base, selected_pair, selected_layer)

def handle_mixed_model_view(model_name, model_base, selected_pair, selected_layer, available_layers):
    """Handle view for mixed models"""
    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["Cluster View", "Predefined Keywords", "Token Search"])
    
    with tab1:
        display_mixed_clusters(model_name, model_base, selected_pair, selected_layer)
    
    with tab2:
        add_predefined_keywords_tab(model_name, model_base, selected_pair, available_layers)
        
    with tab3:
        handle_token_search(model_name, model_base, selected_pair, available_layers)

def handle_standard_model_view(model_name, model_base, selected_pair, selected_layer):
    """Handle view logic for standard models"""
    view = st.sidebar.radio(
        "View", 
        ["Individual Clusters", "Aligned Clusters", "Top Semantic Tags"]
    )

    if view == "Top Semantic Tags":
        display_top_semantic_tags(model_base, selected_pair)
    elif view == "Aligned Clusters":
        display_aligned_clusters(model_base, selected_pair, selected_layer)
    else:  # Individual Clusters
        component = st.sidebar.radio("Component", ["encoder", "decoder"])
        display_standard_clusters(model_name, model_base, selected_pair, selected_layer, component)

if __name__ == "__main__":
    main()