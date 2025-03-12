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
import numpy as np
import gdown

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
    
    # Add radio button for visualization type
    if not model_base.startswith(("coderosetta_mlm_mixed", "coderosetta_aer_mixed")):
        view_type = st.radio(
            "Select View",
            ["Cluster Analysis", "Alignment Metrics"],
            key="view_type"
        )
    else:
        view_type = st.radio(
            "Select View",
            ["Cluster Analysis", "Language Distribution"],
            key="view_type"
        )

    if view_type == "Alignment Metrics":
        display_layer_alignment_metrics(model_base, selected_pair, [selected_layer])
        return
        
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
    
    if not os.path.exists(metrics_file):
        st.error("No alignment data found for this layer")
        return
        
    with open(metrics_file, 'r') as f:
        alignment_metrics = json.load(f)

    # Initialize alignments data structure
    alignments = {"alignments": {}}
    
    # Try to load rich alignment data, fall back to metrics-only if not available
    if os.path.exists(alignments_file):
        with open(alignments_file, 'r') as f:
            alignments = json.load(f)
    else:
        # Create basic alignments structure from metrics file
        for src_cluster_id, data in alignment_metrics.items():
            alignments["alignments"][src_cluster_id] = {
                "encoder_cluster": {
                    "id": f"c{src_cluster_id}",
                    "syntactic_label": "N/A",
                    "semantic_tags": [],
                    "description": "Alignment information from metrics only",
                    "unique_tokens": []  # Will be populated from sentences later
                },
                "aligned_decoder_clusters": [
                    {
                        "id": f"c{target_id}",
                        "syntactic_label": "N/A",
                        "semantic_tags": [],
                        "description": f"Match: {data['metrics'].get('match_percentage', 'N/A')}, Align: {data['metrics'].get('calign_score', 'N/A'):.2%}, Overlap: {data['metrics'].get('colap_score', 'N/A'):.2%}",
                        "unique_tokens": []  # Will be populated from sentences later
                    }
                    for target_id in data.get("aligned_clusters", [])
                ]
            }

    # Create dropdown options for cluster pairs
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
        format_func=lambda x: f"Source {cluster_pairs[x][0]} → Target {cluster_pairs[x][1]}",
        index=st.session_state.current_cluster_index
    )
    
    # Get the selected encoder and decoder IDs
    selected_encoder_id, selected_decoder_id = cluster_pairs[selected_pair_idx]
    
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
    
    # Populate unique tokens if not already present
    if not encoder_cluster.get('unique_tokens') and encoder_sentences:
        encoder_cluster['unique_tokens'] = list(set(
            sent_info["token"] for sent_info in encoder_sentences.get(encoder_cluster['id'], [])
        ))
    
    if not decoder_cluster.get('unique_tokens') and decoder_sentences:
        decoder_cluster['unique_tokens'] = list(set(
            sent_info["token"] for sent_info in decoder_sentences.get(decoder_cluster['id'], [])
        ))
    
    # Display clusters side by side with wordclouds
    st.write("### Cluster Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Source Cluster")
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
        st.write("#### Target Cluster")
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
        st.write("### Source Context Sentences")
        for sent_info in encoder_sentences[encoder_cluster['id']]:
            tokens = sent_info["sentence"].split()
            html = create_sentence_html(tokens, sent_info)
            st.markdown(html, unsafe_allow_html=True)
            
    if decoder_sentences.get(decoder_cluster['id']):
        st.write("### Target Context Sentences")
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
    """Get list of available layers for the selected model and language pair."""
    pair_dir = os.path.join(model_base, selected_pair)
    layers = []
    
    if os.path.exists(pair_dir):
        print(f"Scanning directory: {pair_dir}")  # Debug print
        for item in os.listdir(pair_dir):
            print(f"Found item: {item}")  # Debug print
            # Only process directories that start with 'layer'
            if item.startswith('layer'):
                try:
                    layer_num = int(item.replace('layer', ''))
                    print(f"Adding layer: {layer_num}")  # Debug print
                    layers.append(layer_num)
                except ValueError:
                    print(f"Skipping invalid layer format: {item}")  # Debug print
                    continue
    else:
        print(f"Directory not found: {pair_dir}")  # Debug print
    
    sorted_layers = sorted(layers)
    print(f"Final layers list: {sorted_layers}")  # Debug print
    return sorted_layers

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
    """Count clusters dominated by each language using proportional thresholds."""
    stats = {
        "cpp_dominated": 0,
        "cuda_dominated": 0,
        "mixed": 0,
        "total": 0,
        "small_clusters": 0,
        "detailed_stats": [],
        "diversity_summary": {
            "cpp_dominated": {
                "total_unique_tokens": 0, 
                "cluster_count": 0,
                "max_unique_tokens": 0,
                "min_unique_tokens": float('inf')
            },
            "cuda_dominated": {
                "total_unique_tokens": 0, 
                "cluster_count": 0,
                "max_unique_tokens": 0,
                "min_unique_tokens": float('inf')
            },
            "mixed": {
                "total_unique_tokens": 0, 
                "cluster_count": 0,
                "max_unique_tokens": 0,
                "min_unique_tokens": float('inf')
            }
        }
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
            
            if total_tokens < min_tokens:
                stats["small_clusters"] += 1
                continue
                
            cpp_prop = lang_stats["cpp_count"] / total_tokens
            cuda_prop = lang_stats["cuda_count"] / total_tokens
            mixed_prop = lang_stats["mixed_count"] / total_tokens
            
            # Count unique tokens
            unique_tokens = len(set(sent_info["token"] for sent_info in sentences_list))
            
            cluster_detail = {
                "cluster_id": cluster_id,
                "total_tokens": total_tokens,
                "unique_tokens": unique_tokens,
                "cpp_proportion": cpp_prop,
                "cuda_proportion": cuda_prop,
                "mixed_proportion": mixed_prop
            }
            
            # Determine classification using more nuanced criteria
            category = None
            if cpp_prop >= dominance_threshold:
                cluster_detail["classification"] = "cpp"
                stats["cpp_dominated"] += 1
                category = "cpp_dominated"
            elif cuda_prop >= dominance_threshold:
                cluster_detail["classification"] = "cuda"
                stats["cuda_dominated"] += 1
                category = "cuda_dominated"
            else:
                if mixed_prop > 0.3:
                    cluster_detail["classification"] = "truly_mixed"
                elif abs(cpp_prop - cuda_prop) < 0.2:
                    cluster_detail["classification"] = "balanced"
                elif cpp_prop > cuda_prop:
                    cluster_detail["classification"] = "cpp_leaning"
                else:
                    cluster_detail["classification"] = "cuda_leaning"
                stats["mixed"] += 1
                category = "mixed"
            
            # Update diversity statistics
            if category:
                stats["diversity_summary"][category]["total_unique_tokens"] += unique_tokens
                stats["diversity_summary"][category]["cluster_count"] += 1
                stats["diversity_summary"][category]["max_unique_tokens"] = max(
                    stats["diversity_summary"][category]["max_unique_tokens"],
                    unique_tokens
                )
                stats["diversity_summary"][category]["min_unique_tokens"] = min(
                    stats["diversity_summary"][category]["min_unique_tokens"],
                    unique_tokens
                )
            
            stats["detailed_stats"].append(cluster_detail)
            stats["total"] += 1
    
    # Calculate mean unique tokens and handle empty categories
    for category in stats["diversity_summary"]:
        if stats["diversity_summary"][category]["cluster_count"] > 0:
            stats["diversity_summary"][category]["mean_unique_tokens"] = (
                stats["diversity_summary"][category]["total_unique_tokens"] / 
                stats["diversity_summary"][category]["cluster_count"]
            )
            # Convert inf to 0 for min_unique_tokens if no clusters were found
            if stats["diversity_summary"][category]["min_unique_tokens"] == float('inf'):
                stats["diversity_summary"][category]["min_unique_tokens"] = 0
        else:
            stats["diversity_summary"][category]["mean_unique_tokens"] = 0
            stats["diversity_summary"][category]["min_unique_tokens"] = 0
            stats["diversity_summary"][category]["max_unique_tokens"] = 0
    
    return stats

def display_language_distribution(model_base, selected_pair, available_layers):
    """Display enhanced language distribution statistics"""
    # Initialize session state for graph settings if not exists
    if 'graph_settings' not in st.session_state:
        st.session_state.graph_settings = {
            'show_small_clusters': True,
            'show_percentages': False,
            'chart_height': 600,
            'selected_metrics': ['C++ Dominated', 'CUDA Dominated', 'Mixed', 'Small Clusters'],
            'color_scheme': {
                'C++ Dominated': '#90EE90',
                'CUDA Dominated': '#87CEEB',
                'Mixed': '#DDA0DD',
                'Small Clusters': '#808080'
            }
        }

    # Add graph controls in an expander
    with st.expander("Graph Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.graph_settings['show_small_clusters'] = st.checkbox(
                "Show Small Clusters",
                value=st.session_state.graph_settings['show_small_clusters'],
                key='show_small_clusters'
            )
            st.session_state.graph_settings['show_percentages'] = st.checkbox(
                "Show as Percentages",
                value=st.session_state.graph_settings['show_percentages'],
                key='show_percentages'
            )
        with col2:
            st.session_state.graph_settings['chart_height'] = st.slider(
                "Chart Height",
                min_value=400,
                max_value=1000,
                value=st.session_state.graph_settings['chart_height'],
                step=50,
                key='chart_height'
            )
            
        # Multi-select for metrics to display
        st.session_state.graph_settings['selected_metrics'] = st.multiselect(
            "Select Metrics to Display",
            ['C++ Dominated', 'CUDA Dominated', 'Mixed', 'Small Clusters'],
            default=st.session_state.graph_settings['selected_metrics'],
            key='selected_metrics'
        )

    # Rest of your existing controls
    col1, col2 = st.columns(2)
    with col1:
        dominance_threshold = st.slider(
            "Dominance Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.75,
            step=0.05,
            help="Proportion of tokens needed to consider a cluster dominated by a language",
            key="dominance_threshold"
        )
    with col2:
        min_tokens = st.slider(
            "Minimum Tokens",
            min_value=3,
            max_value=20,
            value=8,
            help="Minimum number of tokens needed for reliable classification",
            key="min_tokens"
        )

    # Initialize session state for caching if not exists
    if 'layer_stats_cache' not in st.session_state:
        st.session_state.layer_stats_cache = {}
    if 'last_threshold' not in st.session_state:
        st.session_state.last_threshold = None
    if 'last_min_tokens' not in st.session_state:
        st.session_state.last_min_tokens = None

    # Check if we need to recompute stats
    recompute = (st.session_state.last_threshold != dominance_threshold or 
                 st.session_state.last_min_tokens != min_tokens)

    # Collect all layer statistics only if needed
    if recompute:
        with st.spinner("Computing language distribution statistics..."):
            all_layer_stats = {}
            for layer in available_layers:
                all_layer_stats[layer] = count_language_dominated_clusters(
                    model_base, 
                    selected_pair, 
                    layer,
                    dominance_threshold,
                    min_tokens
                )
            # Update cache and last parameters
            st.session_state.layer_stats_cache = all_layer_stats
            st.session_state.last_threshold = dominance_threshold
            st.session_state.last_min_tokens = min_tokens
    else:
        all_layer_stats = st.session_state.layer_stats_cache

    # Create DataFrame for detailed analysis before creating tabs
    detailed_data = []
    for layer, stats in all_layer_stats.items():
        for cluster in stats['detailed_stats']:
            detailed_data.append({
                'Layer': layer,
                'Cluster ID': cluster['cluster_id'],
                'Classification': cluster['classification'],
                'Total Tokens': cluster['total_tokens'],
                'Unique Tokens': cluster['unique_tokens'],
                'C++ %': cluster['cpp_proportion'] * 100,
                'CUDA %': cluster['cuda_proportion'] * 100,
                'Mixed %': cluster['mixed_proportion'] * 100
            })
    
    df = pd.DataFrame(detailed_data)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Summary View", "Detailed View", "Layerwise Graph", "Cluster Browser"])

    # Now proceed with the rest of the function using the properly initialized DataFrame
    current_params = {
        'threshold': dominance_threshold,
        'min_tokens': min_tokens,
        'layers': sorted(df['Layer'].unique()),
        'classifications': sorted(df['Classification'].unique())
    }

    with tab1:
        for layer, stats in all_layer_stats.items():
            st.write(f"#### Layer {layer}")
            
            # Calculate percentages
            total_valid_clusters = stats['cpp_dominated'] + stats['cuda_dominated'] + stats['mixed']
            if total_valid_clusters > 0:
                cpp_percent = (stats['cpp_dominated'] / total_valid_clusters) * 100
                cuda_percent = (stats['cuda_dominated'] / total_valid_clusters) * 100
                mixed_percent = (stats['mixed'] / total_valid_clusters) * 100
                small_percent = (stats['small_clusters'] / (total_valid_clusters + stats['small_clusters'])) * 100
            else:
                cpp_percent = cuda_percent = mixed_percent = small_percent = 0
            
            # Display language distribution metrics
            cols = st.columns(5)
            with cols[0]:
                st.metric("C++ Dominated", f"{stats['cpp_dominated']} ({cpp_percent:.1f}%)")
            with cols[1]:
                st.metric("CUDA Dominated", f"{stats['cuda_dominated']} ({cuda_percent:.1f}%)")
            with cols[2]:
                st.metric("Mixed", f"{stats['mixed']} ({mixed_percent:.1f}%)")
            with cols[3]:
                st.metric("Total Clusters", stats['total'])
            with cols[4]:
                st.metric("Small Clusters", f"{stats['small_clusters']} ({small_percent:.1f}%)")
            
            # Add token diversity statistics
            st.write("##### Token Diversity Statistics")
            div_cols = st.columns(3)
            
            with div_cols[0]:
                st.write("**C++ Dominated Clusters**")
                st.write(f"Mean unique tokens: {stats['diversity_summary']['cpp_dominated']['mean_unique_tokens']:.1f}")
                st.write(f"Max unique tokens: {stats['diversity_summary']['cpp_dominated']['max_unique_tokens']}")
                st.write(f"Min unique tokens: {stats['diversity_summary']['cpp_dominated']['min_unique_tokens']}")
            
            with div_cols[1]:
                st.write("**CUDA Dominated Clusters**")
                st.write(f"Mean unique tokens: {stats['diversity_summary']['cuda_dominated']['mean_unique_tokens']:.1f}")
                st.write(f"Max unique tokens: {stats['diversity_summary']['cuda_dominated']['max_unique_tokens']}")
                st.write(f"Min unique tokens: {stats['diversity_summary']['cuda_dominated']['min_unique_tokens']}")
            
            with div_cols[2]:
                st.write("**Mixed Clusters**")
                st.write(f"Mean unique tokens: {stats['diversity_summary']['mixed']['mean_unique_tokens']:.1f}")
                st.write(f"Max unique tokens: {stats['diversity_summary']['mixed']['max_unique_tokens']}")
                st.write(f"Min unique tokens: {stats['diversity_summary']['mixed']['min_unique_tokens']}")
    
    with tab2:
        st.write("### Detailed Cluster Analysis")
        
        # Initialize session state for detailed view if not exists
        if 'detailed_data_cache' not in st.session_state:
            st.session_state.detailed_data_cache = None
        if 'last_detailed_params' not in st.session_state:
            st.session_state.last_detailed_params = None
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_layers = st.multiselect(
                "Filter by Layer",
                options=sorted(df['Layer'].unique()),
                default=sorted(df['Layer'].unique())
            )
            current_params['selected_layers'] = selected_layers
        with col2:
            selected_classifications = st.multiselect(
                "Filter by Classification",
                options=sorted(df['Classification'].unique()),
                default=sorted(df['Classification'].unique())
            )
            current_params['selected_classifications'] = selected_classifications
        with col3:
            min_unique_tokens = st.number_input(
                "Minimum Unique Tokens",
                min_value=0,
                value=0
            )
            current_params['min_unique_tokens'] = min_unique_tokens
        
        # Check if we need to recompute detailed data
        recompute_detailed = (
            st.session_state.detailed_data_cache is None or
            st.session_state.last_detailed_params != current_params
        )
        
        if recompute_detailed:
            with st.spinner("Computing detailed cluster analysis..."):
                detailed_data = []
                for layer, stats in all_layer_stats.items():
                    for cluster in stats['detailed_stats']:
                        detailed_data.append({
                            'Layer': layer,
                            'Cluster ID': cluster['cluster_id'],
                            'Classification': cluster['classification'],
                            'Total Tokens': cluster['total_tokens'],
                            'Unique Tokens': cluster['unique_tokens'],
                            'C++ %': cluster['cpp_proportion'] * 100,
                            'CUDA %': cluster['cuda_proportion'] * 100,
                            'Mixed %': cluster['mixed_proportion'] * 100
                        })
                
                df = pd.DataFrame(detailed_data)
                st.session_state.detailed_data_cache = df
                st.session_state.last_detailed_params = current_params
        else:
            df = st.session_state.detailed_data_cache
        
        # Apply filters
        mask = (
            df['Layer'].isin(selected_layers) &
            df['Classification'].isin(selected_classifications) &
            (df['Unique Tokens'] >= min_unique_tokens)
        )
        filtered_df = df[mask]
        
        # Display filtered DataFrame
        st.dataframe(
            filtered_df.style.format({
                'C++ %': '{:.1f}%',
                'CUDA %': '{:.1f}%',
                'Mixed %': '{:.1f}%'
            }),
            height=400
        )
        
        # Add download button for detailed data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Detailed Data as CSV",
            csv,
            "cluster_analysis.csv",
            "text/csv",
            key='download-csv'
        )
    
    with tab3:
        st.write(f"### Layerwise Distribution (Threshold: {dominance_threshold:.2f}, Min Tokens: {min_tokens})")
        
        # Create the figure with persisted settings
        fig = go.Figure()
        
        # Create data arrays from layer statistics
        layers = []
        cpp_dominated = []
        cuda_dominated = []
        mixed = []
        small_clusters = []
        
        for layer in sorted(all_layer_stats.keys()):
            stats = all_layer_stats[layer]
            layers.append(layer)
            cpp_dominated.append(stats['cpp_dominated'])
            cuda_dominated.append(stats['cuda_dominated'])
            mixed.append(stats['mixed'])
            small_clusters.append(stats['small_clusters'])

        # Now create the metrics_data dictionary
        metrics_data = {
            'C++ Dominated': {'data': cpp_dominated, 'color': st.session_state.graph_settings['color_scheme']['C++ Dominated']},
            'CUDA Dominated': {'data': cuda_dominated, 'color': st.session_state.graph_settings['color_scheme']['CUDA Dominated']},
            'Mixed': {'data': mixed, 'color': st.session_state.graph_settings['color_scheme']['Mixed']},
            'Small Clusters': {'data': small_clusters, 'color': st.session_state.graph_settings['color_scheme']['Small Clusters']}
        }

        for metric_name in st.session_state.graph_settings['selected_metrics']:
            if metric_name in metrics_data:
                data = metrics_data[metric_name]['data']
                if st.session_state.graph_settings['show_percentages']:
                    total = [sum(x) for x in zip(cpp_dominated, cuda_dominated, mixed, small_clusters)]
                    data = [d/t*100 if t > 0 else 0 for d, t in zip(data, total)]
                
                fig.add_trace(go.Scatter(
                    x=layers,
                    y=data,
                    name=metric_name,
                    mode='lines+markers',
                    line=dict(color=metrics_data[metric_name]['color'], width=2),
                    marker=dict(size=8)
                ))

        # Update layout with persisted settings
        fig.update_layout(
            title=dict(
                text=f'Language Distribution Across Layers - {model_base}<br><sup>Dominance Threshold: {dominance_threshold:.2f}, Minimum Tokens: {min_tokens}</sup>',
                font=dict(weight='bold', size=20),
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis_title=dict(
                text='Layer',
                font=dict(weight='bold', size=14)
            ),
            yaxis_title=dict(
                text='Percentage' if st.session_state.graph_settings['show_percentages'] else 'Count',
                font=dict(weight='bold', size=14)
            ),
            hovermode='x unified',
            height=st.session_state.graph_settings['chart_height'],
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

        # Create nuanced classification data
        nuanced_data = {
            'layers': layers,
            'cpp_leaning': [],
            'cuda_leaning': [],
            'truly_mixed': [],
            'balanced': []
        }
        
        # Populate nuanced data arrays
        for layer in sorted(all_layer_stats.keys()):
            stats = all_layer_stats[layer]
            
            cpp_leaning = 0
            cuda_leaning = 0
            truly_mixed = 0
            balanced = 0
            
            for cluster in stats['detailed_stats']:
                if cluster['classification'] == 'cpp_leaning':
                    cpp_leaning += 1
                elif cluster['classification'] == 'cuda_leaning':
                    cuda_leaning += 1
                elif cluster['classification'] == 'truly_mixed':
                    truly_mixed += 1
                elif cluster['classification'] == 'balanced':
                    balanced += 1
            
            nuanced_data['cpp_leaning'].append(cpp_leaning)
            nuanced_data['cuda_leaning'].append(cuda_leaning)
            nuanced_data['truly_mixed'].append(truly_mixed)
            nuanced_data['balanced'].append(balanced)

        # Create nuanced classification figure
        fig_nuanced = go.Figure()
        
        colors = {
            'cpp_leaning': '#98FB98',
            'cuda_leaning': '#ADD8E6',
            'truly_mixed': '#DDA0DD',
            'balanced': '#F0E68C'
        }

        # Add traces with percentage option
        for classification, color in colors.items():
            if nuanced_data[classification]:  # Now this check will work
                data = nuanced_data[classification]
                if st.session_state.graph_settings['show_percentages']:
                    total = [sum(x) for x in zip(*[nuanced_data[c] for c in colors.keys() if nuanced_data[c]])]
                    data = [d/t*100 if t > 0 else 0 for d, t in zip(data, total)]
                
                fig_nuanced.add_trace(go.Scatter(
                    x=nuanced_data['layers'],
                    y=data,
                    name=classification.replace('_', ' ').title(),
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=8)
                ))

        # Update nuanced layout with persisted settings
        fig_nuanced.update_layout(
            title=dict(
                text=f'Intermediate Classification Distribution - {model_base}<br><sup>Dominance Threshold: {dominance_threshold:.2f}, Minimum Tokens: {min_tokens}</sup>',
                font=dict(weight='bold', size=20),
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis_title=dict(
                text='Layer',
                font=dict(weight='bold', size=14)
            ),
            yaxis_title=dict(
                text='Percentage' if st.session_state.graph_settings['show_percentages'] else 'Count',
                font=dict(weight='bold', size=14)
            ),
            hovermode='x unified',
            height=st.session_state.graph_settings['chart_height'],
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(weight='bold')
            )
        )

        fig_nuanced.update_xaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
        fig_nuanced.update_yaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')

        st.plotly_chart(fig_nuanced, use_container_width=True)

    with tab4:
        st.write("### Cluster Browser")
        
        # Layer selection for cluster browser
        selected_layer = st.selectbox(
            "Select Layer",
            available_layers,
            format_func=lambda x: f"Layer {x}",
            key="cluster_browser_layer"
        )
        
        # Get clusters for selected layer
        layer_stats = count_language_dominated_clusters(
            model_base, 
            selected_pair, 
            selected_layer,
            dominance_threshold,
            min_tokens
        )
        
        # Create category selection
        category = st.selectbox(
            "Select Category",
            ["C++ Dominated", "CUDA Dominated", "Mixed"],
            key="cluster_category"
        )
        
        # Filter clusters based on category
        filtered_clusters = []
        for cluster_detail in layer_stats['detailed_stats']:
            if (category == "C++ Dominated" and cluster_detail['classification'] == "cpp") or \
               (category == "CUDA Dominated" and cluster_detail['classification'] == "cuda") or \
               (category == "Mixed" and cluster_detail['classification'] in 
                ["truly_mixed", "balanced", "cpp_leaning", "cuda_leaning"]):
                filtered_clusters.append(cluster_detail)
        
        # Sort clusters by unique tokens instead of total tokens
        filtered_clusters.sort(key=lambda x: x['unique_tokens'], reverse=True)
        
        # Display cluster selection
        if filtered_clusters:
            cluster_options = [f"Cluster {c['cluster_id']} ({c['classification']}, {c['unique_tokens']} unique tokens)" 
                             for c in filtered_clusters]
            selected_cluster = st.selectbox(
                "Select Cluster",
                range(len(cluster_options)),
                format_func=lambda i: cluster_options[i],
                key="cluster_browser_select"
            )
            
            # Display selected cluster details
            if selected_cluster is not None:
                cluster_detail = filtered_clusters[selected_cluster]
                st.write(f"#### Cluster {cluster_detail['cluster_id']} Details")
                
                # Load cluster sentences to get unique tokens
                cluster_sentences = load_cluster_sentences(
                    os.path.join(model_base, selected_pair),
                    selected_layer,
                    "mixed"
                )
                
                if cluster_sentences and cluster_detail['cluster_id'] in cluster_sentences:
                    sentences_data = cluster_sentences[cluster_detail['cluster_id']]
                    if isinstance(sentences_data, dict):
                        unique_tokens = sentences_data.get('unique_tokens', [])
                    else:
                        unique_tokens = list({sent_info["token"] for sent_info in sentences_data})
                    
                    # Create metrics with unique tokens count
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Unique Tokens", len(unique_tokens))
                    with col2:
                        st.metric("C++ Proportion", f"{cluster_detail['cpp_proportion']:.1%}")
                    with col3:
                        st.metric("CUDA Proportion", f"{cluster_detail['cuda_proportion']:.1%}")
                    
                    # Generate and display word cloud for unique tokens
                    if unique_tokens:
                        st.write("#### Word Cloud of Unique Tokens")
                        wc = create_wordcloud(unique_tokens)
                        if wc:
                            fig = plt.figure(figsize=(10, 5))
                            plt.imshow(wc, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    # Display sentences with language tabs
                    st.write("#### Context Sentences")
                    context_sentences = sentences_data.get('sentences', sentences_data)
                    stats = get_language_statistics(context_sentences, os.path.join(model_base, selected_pair))
                    
                    if stats:
                        sent_tab1, sent_tab2, sent_tab3, sent_tab4 = st.tabs(["C++", "CUDA", "Mixed", "Unknown"])
                        
                        with sent_tab1:
                            if stats["cpp_sentences"]:
                                for token, sentence in stats["cpp_sentences"]:
                                    html = create_sentence_html(sentence.split(), {"sentence": sentence, "token": token})
                                    st.markdown(html, unsafe_allow_html=True)
                            else:
                                st.write("No C++ sentences found")
                        
                        with sent_tab2:
                            if stats["cuda_sentences"]:
                                for token, sentence in stats["cuda_sentences"]:
                                    html = create_sentence_html(sentence.split(), {"sentence": sentence, "token": token})
                                    st.markdown(html, unsafe_allow_html=True)
                            else:
                                st.write("No CUDA sentences found")
                        
                        with sent_tab3:
                            if stats["mixed_sentences"]:
                                for token, sentence in stats["mixed_sentences"]:
                                    html = create_sentence_html(sentence.split(), {"sentence": sentence, "token": token})
                                    st.markdown(html, unsafe_allow_html=True)
                            else:
                                st.write("No mixed sentences found")
                        
                        with sent_tab4:
                            if stats["unknown_sentences"]:
                                for token, sentence in stats["unknown_sentences"]:
                                    html = create_sentence_html(sentence.split(), {"sentence": sentence, "token": token})
                                    st.markdown(html, unsafe_allow_html=True)
                            else:
                                st.write("No unknown sentences found")
        else:
            st.warning(f"No clusters found for category: {category}")

def find_clusters_for_token_standard(model_base: str, selected_pair: str, selected_layer: int, search_token: str, component: str):
    """Find all clusters containing the specified token for standard (non-mixed) models"""
    cluster_file = os.path.join(
        model_base, 
        selected_pair,
        f"layer{selected_layer}",
        f"{component}-clusters-kmeans-500.txt"
    )
    
    # Dictionary to store clusters containing the token
    token_clusters = {}
    
    # Check if file exists (especially important for coconet_codebert which doesn't have decoder files)
    if not os.path.exists(cluster_file):
        return token_clusters
    
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|||')
            if len(parts) == 5:  # token|||other|||sent_id|||token_idx|||cluster_id
                token = parts[0].strip()
                cluster_id = parts[4].strip()
                
                if search_token.lower() in token.lower():
                    if f"c{cluster_id}" not in token_clusters:
                        token_clusters[f"c{cluster_id}"] = {
                            'matching_tokens': set(),
                            'all_tokens': set()
                        }
                    token_clusters[f"c{cluster_id}"]['matching_tokens'].add(token)
                    token_clusters[f"c{cluster_id}"]['all_tokens'].add(token)
                elif f"c{cluster_id}" in token_clusters:
                    token_clusters[f"c{cluster_id}"]['all_tokens'].add(token)
    
    return token_clusters

def display_search_results_standard(model_name, model_base, selected_pair, selected_layer, search_token, search_token2=None):
    """Display search results for tokens in standard (non-mixed) models"""
    st.write(f"### Token Search")
    
    # Initialize state for search results if not exists
    if 'search_results_state' not in st.session_state:
        st.session_state.search_results_state = {
            'encoder_results': {},
            'decoder_results': {},
            'matching_tokens': set(),
            'matching_tokens2': set(),
            'last_search': None,
            'last_search2': None
        }
    
    # Check if we need to recompute results
    if (search_token != st.session_state.search_results_state['last_search'] or 
        search_token2 != st.session_state.search_results_state['last_search2']):
        
        # Find all matching tokens across components
        all_matching_tokens = set()
        all_matching_tokens2 = set()
        encoder_results = {}
        decoder_results = {}
        
        # If second token provided, search for co-occurrences
        if search_token2 and search_token2.strip():
            for component in ['encoder', 'decoder']:
                results = find_clusters_with_multiple_tokens(
                    model_base,
                    selected_pair,
                    selected_layer,
                    [search_token, search_token2],
                    component
                )
                if component == 'encoder':
                    encoder_results = results
                else:
                    decoder_results = results
                
                for cluster_data in results.values():
                    all_matching_tokens.update(cluster_data['matching_tokens'][search_token])
                    all_matching_tokens2.update(cluster_data['matching_tokens'][search_token2])
        else:
            # Single token search
            for component in ['encoder', 'decoder']:
                results = find_clusters_for_token_standard(
                    model_base, 
                    selected_pair, 
                    selected_layer, 
                    search_token,
                    component
                )
                if component == 'encoder':
                    encoder_results = results
                else:
                    decoder_results = results
                    
                for cluster_data in results.values():
                    all_matching_tokens.update(cluster_data['matching_tokens'])
        
        # Update state
        st.session_state.search_results_state.update({
            'encoder_results': encoder_results,
            'decoder_results': decoder_results,
            'matching_tokens': sorted(all_matching_tokens),
            'matching_tokens2': sorted(all_matching_tokens2),
            'last_search': search_token,
            'last_search2': search_token2
        })
    
    matching_tokens_list = st.session_state.search_results_state['matching_tokens']
    
    if not matching_tokens_list:
        st.warning("No matching tokens found")
        return
    
    # Display token selection
    selected_token = st.selectbox(
        "Select token:",
        matching_tokens_list,
        key="token_selector_standard_1"
    )
    
    if search_token2 and search_token2.strip():
        matching_tokens_list2 = st.session_state.search_results_state['matching_tokens2']
        if matching_tokens_list2:
            selected_token2 = st.selectbox(
                "Select second token:",
                matching_tokens_list2,
                key="token_selector_standard_2"
            )
        else:
            selected_token2 = None
            st.warning("No matching tokens found for second search term")
    else:
        selected_token2 = None

    # Display results based on selected tokens
    if selected_token:
        # Create tabs for encoder and decoder results
        tab1, tab2 = st.tabs(["Source (Encoder)", "Target (Decoder)"])
        
        with tab1:
            st.write("### Source Clusters")
            encoder_results = st.session_state.search_results_state['encoder_results']
            
            if encoder_results:
                for cluster_id, cluster_data in encoder_results.items():
                    with st.expander(f"Cluster {cluster_id}"):
                        # Display token information
                        st.write("**Matching Tokens:**")
                        if selected_token2:
                            st.write(f"Token 1 matches: {', '.join(cluster_data['matching_tokens'][search_token])}")
                            st.write(f"Token 2 matches: {', '.join(cluster_data['matching_tokens'][search_token2])}")
                        else:
                            if isinstance(cluster_data['matching_tokens'], dict):
                                st.write(f"Matches: {', '.join(cluster_data['matching_tokens'][search_token])}")
                            else:
                                st.write(f"Matches: {', '.join(cluster_data['matching_tokens'])}")
                        
                        st.write("**All Tokens in Cluster:**")
                        st.write(", ".join(sorted(cluster_data['all_tokens'])))
                        
                        # Load and display cluster sentences
                        sentences = load_cluster_sentences(
                            os.path.join(model_base, selected_pair),
                            selected_layer,
                            "encoder"
                        )
                        
                        if sentences and cluster_id in sentences:
                            st.write("**Context Sentences:**")
                            for sent_info in sentences[cluster_id]:
                                html = create_sentence_html(sent_info["sentence"].split(), sent_info)
                                st.markdown(html, unsafe_allow_html=True)
                        
                        # Display Gemini labels
                        display_individual_cluster_labels(
                            cluster_id,
                            model_base,
                            selected_pair,
                            selected_layer,
                            "encoder"
                        )
            else:
                st.info("No matching source clusters found")
        
        with tab2:
            st.write("### Target Clusters")
            decoder_results = st.session_state.search_results_state['decoder_results']
            
            if decoder_results:
                for cluster_id, cluster_data in decoder_results.items():
                    with st.expander(f"Cluster {cluster_id}"):
                        # Display token information
                        st.write("**Matching Tokens:**")
                        if selected_token2:
                            st.write(f"Token 1 matches: {', '.join(cluster_data['matching_tokens'][search_token])}")
                            st.write(f"Token 2 matches: {', '.join(cluster_data['matching_tokens'][search_token2])}")
                        else:
                            if isinstance(cluster_data['matching_tokens'], dict):
                                st.write(f"Matches: {', '.join(cluster_data['matching_tokens'][search_token])}")
                            else:
                                st.write(f"Matches: {', '.join(cluster_data['matching_tokens'])}")
                        
                        st.write("**All Tokens in Cluster:**")
                        st.write(", ".join(sorted(cluster_data['all_tokens'])))
                        
                        # Load and display cluster sentences
                        sentences = load_cluster_sentences(
                            os.path.join(model_base, selected_pair),
                            selected_layer,
                            "decoder"
                        )
                        
                        if sentences and cluster_id in sentences:
                            st.write("**Context Sentences:**")
                            for sent_info in sentences[cluster_id]:
                                html = create_sentence_html(sent_info["sentence"].split(), sent_info)
                                st.markdown(html, unsafe_allow_html=True)
                        
                        # Display Gemini labels
                        display_individual_cluster_labels(
                            cluster_id,
                            model_base,
                            selected_pair,
                            selected_layer,
                            "decoder"
                        )
            else:
                st.info("No matching target clusters found")

def find_matching_tokens(model_base: str, selected_pair: str, layer: int, search_term: str) -> List[str]:
    """Find all tokens that match the search term"""
    matching_tokens = set()
    
    # Check both encoder and decoder files
    for component in ['encoder', 'decoder']:
        cluster_file = os.path.join(
            model_base, 
            selected_pair,
            f"layer{layer}",
            f"{component}-clusters-kmeans-500.txt"
        )
        if os.path.exists(cluster_file):
            with open(cluster_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|||')
                    if len(parts) >= 1:
                        token = parts[0].strip()
                        if search_term.lower() in token.lower():
                            matching_tokens.add(token)
    
    return sorted(matching_tokens)

def find_matching_semantic_tags(model_base: str, selected_pair: str, layer: int, search_term: str) -> List[str]:
    """Find all semantic tags that match the search term"""
    matching_tags = set()
    
    # Check both encoder and decoder label files
    for component in ['encoder', 'decoder']:
        labels_file = Path(model_base) / selected_pair / f"layer{layer}" / f"{component}_gemini_labels.json"
        if labels_file.exists():
            with open(labels_file) as f:
                labels_data = json.load(f)
                for cluster_labels in labels_data:
                    for labels in cluster_labels.values():
                        for tag in labels.get("Semantic Tags", []):
                            if search_term.lower() in tag.lower():
                                matching_tags.add(tag)
    
    return sorted(matching_tags)

def handle_standard_model_view(model_name, model_base, selected_pair, selected_layer):
    """Handle view logic for standard models"""
    # Get all available layers for the model
    available_layers = get_available_layers(model_base, selected_pair)

    if model_name == "coconet_codebert" or model_name == "coconet_deepseek":
        view = st.sidebar.radio(
            "View", 
            ["Search & Analysis", "Individual Clusters", "Token Pairs"]
        )
    
    else:
        view = st.sidebar.radio(
            "View", 
            ["Search & Analysis", "Individual Clusters", "Aligned Clusters", "Semantic Alignments", "Top Semantic Tags", "Token Pairs"]
        )

    if view == "Search & Analysis":
        # Create tabs for different search types
        search_type = st.radio(
            "Search Type",
            ["Token Search", "Token Pair Search", "Semantic Tag Search"],
            horizontal=True
        )
        
        if search_type == "Token Search":
            st.write("## Token Search")
            
            # Token search box with suggestions
            search_term = st.text_input(
                "Search for tokens:",
                help="Type to search for tokens (e.g., 'for', 'if', 'while')"
            )
            
            if search_term:
                matching_tokens = find_matching_tokens(
                    model_base,
                    selected_pair,
                    available_layers[0],
                    search_term
                )
                
                if matching_tokens:
                    selected_token = st.selectbox(
                        "Select a token:",
                        matching_tokens,
                        format_func=lambda x: f"{x} (token)",
                        key="token_selector"
                    )
                    
                    if selected_token and st.button("Analyze Token", type="primary"):
                        with st.spinner(f"Analyzing token '{selected_token}'..."):
                            evolution_data = analyze_token_evolution(
                                model_base,
                                selected_pair,
                                available_layers,
                                [selected_token]
                            )
                            
                            if evolution_data:
                                tab1, tab2 = st.tabs(["Evolution Analysis", "Cluster Details"])
                                
                                with tab1:
                                    display_token_evolution(evolution_data, [selected_token])
                                
                                with tab2:
                                    st.write("### Cluster Analysis")
                                    subtab1, subtab2 = st.tabs(["Source Clusters", "Target Clusters"])
                                    
                                    with subtab1:
                                        display_token_clusters(model_base, selected_pair, available_layers[0], selected_token, "encoder")
                                    
                                    with subtab2:
                                        display_token_clusters(model_base, selected_pair, available_layers[0], selected_token, "decoder")
                            else:
                                st.warning("No data found for this token")
                else:
                    st.info("No matching tokens found")
        
        elif search_type == "Token Pair Search":
            st.write("## Token Pair Search")
            
            # First token search
            search_term1 = st.text_input(
                "Search for first token:",
                help="Type to search for the first token"
            )
            
            if search_term1:
                matching_tokens1 = find_matching_tokens(
                    model_base,
                    selected_pair,
                    available_layers[0],
                    search_term1
                )
                
                if matching_tokens1:
                    selected_token1 = st.selectbox(
                        "Select first token:",
                        matching_tokens1,
                        format_func=lambda x: f"{x} (token)",
                        key="token_pair_selector1"
                    )
                    
                    # Second token search
                    search_term2 = st.text_input(
                        "Search for second token:",
                        help="Type to search for the second token"
                    )
                    
                    if search_term2:
                        matching_tokens2 = find_matching_tokens(
                            model_base,
                            selected_pair,
                            available_layers[0],
                            search_term2
                        )
                        
                        if matching_tokens2:
                            selected_token2 = st.selectbox(
                                "Select second token:",
                                matching_tokens2,
                                format_func=lambda x: f"{x} (token)",
                                key="token_pair_selector2"
                            )
                            
                            if selected_token1 and selected_token2 and st.button("Analyze Token Pair", type="primary"):
                                with st.spinner(f"Analyzing token pair '{selected_token1}' and '{selected_token2}'..."):
                                    evolution_data = analyze_token_evolution(
                                        model_base,
                                        selected_pair,
                                        available_layers,
                                        [selected_token1, selected_token2]
                                    )
                                    
                                    if evolution_data:
                                        tab1, tab2 = st.tabs(["Evolution Analysis", "Co-occurrence Analysis"])
                                        
                                        with tab1:
                                            display_token_evolution(evolution_data, [selected_token1, selected_token2])
                                        
                                        with tab2:
                                            st.write("### Co-occurrence Analysis")
                                            subtab1, subtab2 = st.tabs(["Source Clusters", "Target Clusters"])
                                            
                                            with subtab1:
                                                display_token_pair_clusters(model_base, selected_pair, available_layers[0], selected_token1, selected_token2, "encoder")
                                            
                                            with subtab2:
                                                display_token_pair_clusters(model_base, selected_pair, available_layers[0], selected_token1, selected_token2, "decoder")
                                    else:
                                        st.warning("No data found for this token pair")
                        else:
                            st.info("No matching tokens found for second search")
                else:
                    st.info("No matching tokens found for first search")
        
        else:  # Semantic Tag Search
            st.write("## Semantic Tag Search")
            handle_semantic_tag_search(model_name, model_base, selected_pair, available_layers)
            
    elif view == "Top Semantic Tags":
        display_top_semantic_tags(model_base, selected_pair)
    elif view == "Aligned Clusters":
        display_aligned_clusters(model_base, selected_pair, selected_layer)
    elif view == "Semantic Alignments":
        display_semantic_alignments(model_base, selected_pair, selected_layer)
    elif view == "Token Pairs":
        display_token_pair_analysis(model_name, model_base, selected_pair, available_layers)
    else:  # Individual Clusters
        if model_name != "coconet_codebert" and model_name != "coconet_deepseek":
            component = st.sidebar.radio("Component", ["source", "target"])
            if component == "source":
                component = "encoder"
            else:
                component = "decoder"
        else:
            component = "encoder"
        display_standard_clusters(model_name, model_base, selected_pair, selected_layer, component)

def display_token_clusters(model_base, selected_pair, layer, token, component):
    """Helper function to display clusters for a token"""
    # For coconet_codebert, skip decoder component
    if "coconet_codebert" in model_base or "coconet_deepseek" in model_base and component == "decoder":
        st.info("CodeBERT model doesn't have target (decoder) clusters")
        return
        
    clusters = find_clusters_for_token_standard(
        model_base,
        selected_pair,
        layer,
        token,
        component
    )
    if clusters:
        for cluster_id, tokens in clusters.items():
            with st.expander(f"Cluster {cluster_id}"):
                st.write("**Matching Tokens:**", ", ".join(tokens['matching_tokens']))
                display_individual_cluster_labels(
                    cluster_id,
                    model_base,
                    selected_pair,
                    layer,
                    component
                )
    else:
        st.info(f"No {component} clusters found containing this token")

def display_token_pair_clusters(model_base, selected_pair, layer, token1, token2, component):
    """Helper function to display clusters for a token pair"""
    # For coconet_codebert, skip decoder component
    if "coconet_codebert" in model_base or "coconet_deepseek" in model_base and component == "decoder":
        st.info("CodeBERT model doesn't have target (decoder) clusters")
        return
        
    clusters = find_clusters_with_multiple_tokens(
        model_base,
        selected_pair,
        layer,
        [token1, token2],
        component
    )
    if clusters:
        for cluster_id, data in clusters.items():
            with st.expander(f"Cluster {cluster_id}"):
                st.write(f"**{token1} occurrences:**", ", ".join(data['matching_tokens'][token1]))
                st.write(f"**{token2} occurrences:**", ", ".join(data['matching_tokens'][token2]))
                display_individual_cluster_labels(
                    cluster_id,
                    model_base,
                    selected_pair,
                    layer,
                    component
                )
    else:
        st.info(f"No {component} clusters found containing both tokens")

def display_search_results(model_name, model_base, selected_pair, available_layers, search_token, search_token2=None, evolution_data=None):
    """Display search results for tokens"""
    st.write(f"### Search Results")
    
    # Initialize state for search results if not exists
    if 'search_results_state' not in st.session_state:
        st.session_state.search_results_state = {
            'encoder_results': {},
            'decoder_results': {},
            'matching_tokens': set(),
            'matching_tokens2': set(),
            'last_search': None,
            'last_search2': None
        }
    
    # Check if we need to recompute results
    if (search_token != st.session_state.search_results_state['last_search'] or 
        search_token2 != st.session_state.search_results_state['last_search2']):
        
        # Find all matching tokens across components
        all_matching_tokens = set()
        all_matching_tokens2 = set()
        encoder_results = {}
        decoder_results = {}
        
        # If second token provided, search for co-occurrences
        if search_token2 and search_token2.strip():
            # For encoder component
            results = find_clusters_with_multiple_tokens(
                model_base,
                selected_pair,
                available_layers[0],
                [search_token, search_token2],
                "encoder"
            )
            encoder_results = results
            
            for cluster_data in results.values():
                all_matching_tokens.update(cluster_data['matching_tokens'][search_token])
                all_matching_tokens2.update(cluster_data['matching_tokens'][search_token2])
            
            # For decoder component (skip for coconet_codebert)
            if not "coconet_codebert" in model_base and not "coconet_deepseek" in model_base    :
                results = find_clusters_with_multiple_tokens(
                    model_base,
                    selected_pair,
                    available_layers[0],
                    [search_token, search_token2],
                    "decoder"
                )
                decoder_results = results
                
                for cluster_data in results.values():
                    all_matching_tokens.update(cluster_data['matching_tokens'][search_token])
                    all_matching_tokens2.update(cluster_data['matching_tokens'][search_token2])
        else:
            # Single token search for encoder
            results = find_clusters_for_token_standard(
                model_base, 
                selected_pair, 
                available_layers[0], 
                search_token,
                "encoder"
            )
            encoder_results = results
                
            for cluster_data in results.values():
                all_matching_tokens.update(cluster_data['matching_tokens'])
            
            # Single token search for decoder (skip for coconet_codebert)
            if not "coconet_codebert" in model_base and not "coconet_deepseek" in model_base:
                results = find_clusters_for_token_standard(
                    model_base, 
                    selected_pair, 
                    available_layers[0], 
                    search_token,
                    "decoder"
                )
                decoder_results = results
                    
                for cluster_data in results.values():
                    all_matching_tokens.update(cluster_data['matching_tokens'])
        
        # Update state
        st.session_state.search_results_state.update({
            'encoder_results': encoder_results,
            'decoder_results': decoder_results,
            'matching_tokens': sorted(all_matching_tokens),
            'matching_tokens2': sorted(all_matching_tokens2),
            'last_search': search_token,
            'last_search2': search_token2
        })
    
    # Create tabs for evolution, encoder, and decoder results
    if "coconet_codebert" in model_base or "coconet_deepseek" in model_base:
        tab1, tab2 = st.tabs(["Evolution Analysis", "Source (Encoder)"])
    else:
        tab1, tab2, tab3 = st.tabs(["Evolution Analysis", "Source (Encoder)", "Target (Decoder)"])
    
    with tab1:
        st.write("### Token Evolution Analysis")
        if evolution_data:
            # Display evolution graph
            display_token_evolution(evolution_data, [search_token] + ([search_token2] if search_token2 else []))
            
            # Display statistics table
            st.write("### Evolution Statistics")
            stats_data = {
                'Layer': evolution_data['layers'],
                f"'{search_token}' Clusters": evolution_data['individual_counts'][search_token]
            }
            
            if search_token2:
                stats_data[f"'{search_token2}' Clusters"] = evolution_data['individual_counts'][search_token2]
                stats_data['Co-occurring Clusters'] = evolution_data['combined_counts']
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df)
        else:
            st.error("Evolution data not available")
    
    with tab2:
        st.write("### Source Clusters")
        encoder_results = st.session_state.search_results_state['encoder_results']
        
        if encoder_results:
            for cluster_id, cluster_data in encoder_results.items():
                with st.expander(f"Cluster {cluster_id}"):
                    # Display token information
                    st.write("**Matching Tokens:**")
                    if search_token2:
                        st.write(f"Token 1 matches: {', '.join(cluster_data['matching_tokens'][search_token])}")
                        st.write(f"Token 2 matches: {', '.join(cluster_data['matching_tokens'][search_token2])}")
                    else:
                        if isinstance(cluster_data['matching_tokens'], dict):
                            st.write(f"Matches: {', '.join(cluster_data['matching_tokens'][search_token])}")
                        else:
                            st.write(f"Matches: {', '.join(cluster_data['matching_tokens'])}")
                    
                    st.write("**All Tokens in Cluster:**")
                    st.write(", ".join(sorted(cluster_data['all_tokens'])))
                    
                    # Load and display cluster sentences
                    sentences = load_cluster_sentences(
                        os.path.join(model_base, selected_pair),
                        available_layers[0],
                        "encoder"
                    )
                    
                    if sentences and cluster_id in sentences:
                        st.write("**Context Sentences:**")
                        for sent_info in sentences[cluster_id]:
                            html = create_sentence_html(sent_info["sentence"].split(), sent_info)
                            st.markdown(html, unsafe_allow_html=True)
                    
                    # Display Gemini labels
                    display_individual_cluster_labels(
                        cluster_id,
                        model_base,
                        selected_pair,
                        available_layers[0],
                        "encoder"
                    )
        else:
            st.info("No matching source clusters found")
    
    if not "coconet_codebert"  in model_base and not "coconet_deepseek" in model_base:
        with tab3:
            st.write("### Target Clusters")
            decoder_results = st.session_state.search_results_state['decoder_results']
            
            if decoder_results:
                for cluster_id, cluster_data in decoder_results.items():
                    with st.expander(f"Cluster {cluster_id}"):
                        # Display token information
                        st.write("**Matching Tokens:**")
                        if search_token2:
                            st.write(f"Token 1 matches: {', '.join(cluster_data['matching_tokens'][search_token])}")
                            st.write(f"Token 2 matches: {', '.join(cluster_data['matching_tokens'][search_token2])}")
                        else:
                            if isinstance(cluster_data['matching_tokens'], dict):
                                st.write(f"Matches: {', '.join(cluster_data['matching_tokens'][search_token])}")
                            else:
                                st.write(f"Matches: {', '.join(cluster_data['matching_tokens'])}")
                        
                        st.write("**All Tokens in Cluster:**")
                        st.write(", ".join(sorted(cluster_data['all_tokens'])))
                        
                        # Load and display cluster sentences
                        sentences = load_cluster_sentences(
                            os.path.join(model_base, selected_pair),
                            available_layers[0],
                            "decoder"
                        )
                        
                        if sentences and cluster_id in sentences:
                            st.write("**Context Sentences:**")
                            for sent_info in sentences[cluster_id]:
                                html = create_sentence_html(sent_info["sentence"].split(), sent_info)
                                st.markdown(html, unsafe_allow_html=True)
                        
                        # Display Gemini labels
                        display_individual_cluster_labels(
                            cluster_id,
                            model_base,
                            selected_pair,
                            available_layers[0],
                            "decoder"
                        )
            else:
                st.info("No matching target clusters found")

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
    
    # Move cluster selection to main view
    if component == "encoder":
        component_display = "Source"
    else:
        component_display = "Target"
        
    st.write(f"### {component_display} Cluster Analysis")
    
    # Cluster selection in main view
    selected_cluster = st.selectbox(
        f"Select {component_display} Cluster",
        clusters,
        format_func=lambda x: f"Cluster {x[1:]}",  # Remove 'c' prefix for display
        index=min(st.session_state.current_cluster_index, len(clusters)-1)
    )
    
    if selected_cluster:
        st.write(f"#### Details for {component_display} Cluster {selected_cluster[1:]}")
        
        # Create word cloud from sentences in this cluster
        cluster_sentences = sentences[selected_cluster]
        tokens = set()
        for sent_info in cluster_sentences:
            tokens.add(sent_info["token"])
            
        if tokens:
            st.write("#### Word Cloud")
            wc = create_wordcloud(list(tokens))
            if wc:
                # Reduced figure size from (10, 5) to (8, 4)
                fig = plt.figure(figsize=(8, 4))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)
        
        # Add Gemini labels display after word cloud
        display_individual_cluster_labels(
            selected_cluster,
            model_base,
            selected_pair,
            selected_layer,
            component
        )
        
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
    
    # Move cluster selection to main view
    st.write("### Mixed Cluster Analysis")
    
    # Cluster selection in main view
    selected_cluster = st.selectbox(
        "Select Mixed Cluster",
        clusters,
        format_func=lambda x: f"Cluster {x[1:]}",  # Remove 'c' prefix for display
        index=min(st.session_state.current_cluster_index, len(clusters)-1)
    )
    
    if selected_cluster:
        st.write(f"#### Details for Cluster {selected_cluster[1:]}")
        
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
                # Reduced figure size from (10, 5) to (8, 4)
                fig = plt.figure(figsize=(8, 4))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                
        # Add mixed model Gemini labels display after word cloud
        display_mixed_cluster_labels(
            selected_cluster,
            model_base,
            selected_pair,
            selected_layer
        )
        
        # Display context sentences
        st.write("#### Context Sentences")
        for sent_info in cluster_sentences:
            html = create_sentence_html(sent_info["sentence"].split(), sent_info)
            st.markdown(html, unsafe_allow_html=True)

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
    
    # Update layout with two y-axes
    fig.update_layout(
        title=f"Evolution of '{keyword}' Across Layers",
        xaxis=dict(title='Layer'),
        yaxis=dict(
            title=dict(text='Number of Clusters', font=dict(color='#1f77b4')),
            tickfont=dict(color='#1f77b4')
        ),
        yaxis2=dict(
            title=dict(text='Total Token Occurrences', font=dict(color='#ff7f0e')),
            tickfont=dict(color='#ff7f0e'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        showlegend=True
    )
    
    # Add unique key for the first plotly chart
    st.plotly_chart(fig, use_container_width=True, key=f"evolution_main_{context}_{keyword}_{unique_suffix}")
    
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
    
    # Create heatmap with unique key
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
    
    # Add unique key for the heatmap
    st.plotly_chart(fig_heatmap, use_container_width=True, key=f"evolution_heatmap_{context}_{keyword}_{unique_suffix}")
    
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
    
    # Initialize graph settings in session state if not exists
    if 'predefined_graph_settings' not in st.session_state:
        st.session_state.predefined_graph_settings = {
            'show_percentages': False,
            'chart_height': 800,
            'selected_cuda_keywords': list(cuda_evolution.keys()),
            'selected_cpp_keywords': list(cpp_evolution.keys()),
            'color_scheme': {
                'CUDA': {k: f'rgb({30+i*30}, {144+i*10}, {255-i*20})' for i, k in enumerate(cuda_evolution.keys())},
                'C++': {k: f'rgb({255-i*20}, {144+i*10}, {30+i*30})' for i, k in enumerate(cpp_evolution.keys())}
            }
        }
    
    # Add graph controls in an expander
    with st.expander("Graph Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.predefined_graph_settings['show_percentages'] = st.checkbox(
                "Show as Percentages",
                value=st.session_state.predefined_graph_settings['show_percentages'],
                key='predefined_show_percentages'
            )
        with col2:
            st.session_state.predefined_graph_settings['chart_height'] = st.slider(
                "Chart Height",
                min_value=400,
                max_value=1200,
                value=st.session_state.predefined_graph_settings['chart_height'],
                step=50,
                key='predefined_chart_height'
            )
        
        # Keyword selection
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.predefined_graph_settings['selected_cuda_keywords'] = st.multiselect(
                "Select CUDA Keywords",
                list(cuda_evolution.keys()),
                default=st.session_state.predefined_graph_settings['selected_cuda_keywords'],
                key='selected_cuda_keywords'
            )
        with col2:
            st.session_state.predefined_graph_settings['selected_cpp_keywords'] = st.multiselect(
                "Select C++ Keywords",
                list(cpp_evolution.keys()),
                default=st.session_state.predefined_graph_settings['selected_cpp_keywords'],
                key='selected_cpp_keywords'
            )
    
    tab1, tab2, tab3 = st.tabs(["Combined View", "CUDA Keywords", "C++ Keywords"])
    
    with tab1:
        st.write("### Combined Keywords Evolution")
        
        # Create combined graph
        fig = go.Figure()
        
        # Add CUDA keywords
        for keyword in st.session_state.predefined_graph_settings['selected_cuda_keywords']:
            data = cuda_evolution[keyword]
            if st.session_state.predefined_graph_settings['show_percentages']:
                total = max(data['cluster_counts']) if data['cluster_counts'] else 1
                y_values = [count/total * 100 for count in data['cluster_counts']]
            else:
                y_values = data['cluster_counts']
                
            fig.add_trace(go.Scatter(
                x=data['layers'],
                y=y_values,
                name=f"CUDA: {keyword}",
                mode='lines+markers',
                line=dict(
                    dash='solid',
                    color=st.session_state.predefined_graph_settings['color_scheme']['CUDA'][keyword]
                ),
                marker=dict(size=8)
            ))
            
        # Add C++ keywords
        for keyword in st.session_state.predefined_graph_settings['selected_cpp_keywords']:
            data = cpp_evolution[keyword]
            if st.session_state.predefined_graph_settings['show_percentages']:
                total = max(data['cluster_counts']) if data['cluster_counts'] else 1
                y_values = [count/total * 100 for count in data['cluster_counts']]
            else:
                y_values = data['cluster_counts']
                
            fig.add_trace(go.Scatter(
                x=data['layers'],
                y=y_values,
                name=f"C++: {keyword}",
                mode='lines+markers',
                line=dict(
                    dash='dot',
                    color=st.session_state.predefined_graph_settings['color_scheme']['C++'][keyword]
                ),
                marker=dict(size=8)
            ))
            
        fig.update_layout(
            title="Evolution of Keywords Across Layers",
            xaxis_title="Layer",
            yaxis_title="Percentage" if st.session_state.predefined_graph_settings['show_percentages'] else "Number of Clusters",
            height=st.session_state.predefined_graph_settings['chart_height'],
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
        selected_cuda = st.session_state.predefined_graph_settings['selected_cuda_keywords']
        heatmap_data = []
        for keyword in selected_cuda:
            row = []
            for layer in available_layers:
                count = cuda_evolution[keyword]['cluster_counts'][
                    cuda_evolution[keyword]['layers'].index(layer)
                ]
                if st.session_state.predefined_graph_settings['show_percentages']:
                    total = max(cuda_evolution[keyword]['cluster_counts'])
                    count = (count / total * 100) if total > 0 else 0
                row.append(count)
            heatmap_data.append(row)
            
        fig_cuda = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=available_layers,
            y=selected_cuda,
            colorscale='Viridis',
            colorbar=dict(
                title='Percentage' if st.session_state.predefined_graph_settings['show_percentages'] else 'Clusters'
            )
        ))
        
        fig_cuda.update_layout(
            title="CUDA Keywords Distribution Across Layers",
            xaxis_title="Layer",
            yaxis_title="Keyword",
            height=st.session_state.predefined_graph_settings['chart_height']
        )
        
        st.plotly_chart(fig_cuda, use_container_width=True)
        
        # Individual CUDA keyword graphs
        for keyword in selected_cuda:
            with st.expander(f"Detailed View: {keyword}"):
                display_keyword_evolution(cuda_evolution[keyword], keyword, "predefined")
        
    with tab3:
        st.write("### C++ Keywords Evolution")
        
        # Create heatmap for C++ keywords
        selected_cpp = st.session_state.predefined_graph_settings['selected_cpp_keywords']
        heatmap_data = []
        for keyword in selected_cpp:
            row = []
            for layer in available_layers:
                count = cpp_evolution[keyword]['cluster_counts'][
                    cpp_evolution[keyword]['layers'].index(layer)
                ]
                if st.session_state.predefined_graph_settings['show_percentages']:
                    total = max(cpp_evolution[keyword]['cluster_counts'])
                    count = (count / total * 100) if total > 0 else 0
                row.append(count)
            heatmap_data.append(row)
            
        fig_cpp = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=available_layers,
            y=selected_cpp,
            colorscale='Viridis',
            colorbar=dict(
                title='Percentage' if st.session_state.predefined_graph_settings['show_percentages'] else 'Clusters'
            )
        ))
        
        fig_cpp.update_layout(
            title="C++ Keywords Distribution Across Layers",
            xaxis_title="Layer",
            yaxis_title="Keyword",
            height=st.session_state.predefined_graph_settings['chart_height']
        )
        
        st.plotly_chart(fig_cpp, use_container_width=True)
        
        # Individual C++ keyword graphs
        for keyword in selected_cpp:
            with st.expander(f"Detailed View: {keyword}"):
                display_keyword_evolution(cpp_evolution[keyword], keyword, "predefined")

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

def download_from_drive(file_id, destination):
    """Download a file from Google Drive"""
    if not os.path.exists(destination):
        # Extract file ID from full Google Drive URL if needed
        if 'drive.google.com' in file_id:
            # Handle different Google Drive URL formats
            if '/file/d/' in file_id:
                file_id = file_id.split('/file/d/')[1].split('/')[0]
            elif 'id=' in file_id:
                file_id = file_id.split('id=')[1].split('&')[0]
            else:
                st.error(f"Invalid Google Drive URL format: {file_id}")
                return False

        # Create URL for direct download
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        try:
            # Try downloading with gdown
            success = gdown.download(url, destination, quiet=False)
            if not success:
                st.error("Failed to download file with gdown. Please check the file permissions.")
                return False
        except Exception as e:
            st.error(f"Error downloading file: {e}")
            return False
    return True

def display_semantic_alignments(model_base: str, selected_pair: str, selected_layer: int):
    """Display semantic alignments between encoder and decoder clusters"""
    
    # Add similarity threshold slider
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.82,  # Default value
        step=0.01,
        format="%.2f",
        help="Only show alignments with similarity score above this threshold"
    )
    
    # Load semantic alignments
    alignment_file = Path(model_base) / selected_pair / f"layer{selected_layer}" / "semantic_alignments.json"
    
    # If file doesn't exist locally, download from Drive
    if not alignment_file.exists():
        drive_file_id = "1ghtRAz4egj8Zw4R5zjSEEBDFZN4Cx-CR"
        if not download_from_drive(drive_file_id, str(alignment_file)):
            st.warning("Could not load semantic alignments. Please ensure the file is publicly accessible.")
            return

    with open(alignment_file) as f:
        alignment_data = json.load(f)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Cluster Details", "Alignment Distribution"])
    
    with tab2:
        st.write("### Distribution of Alignments Across Source Clusters")
        
        # Calculate number of alignments per cluster above threshold
        alignment_counts = [
            len([m for m in align['matches'] if m['similarity'] >= similarity_threshold])
            for align in alignment_data['alignments']
        ]
        cluster_ids = [int(align['encoder_id'].lstrip('c')) for align in alignment_data['alignments']]
        
        # Create line plot using plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cluster_ids,
            y=alignment_counts,
            mode='lines',
            name='Number of Alignments'
        ))
        
        # Update x-axis to show ticks at multiples of 50
        fig.update_layout(
            title=f"Number of Alignments per Source Cluster (Similarity ≥ {similarity_threshold:.2%})",
            xaxis=dict(
                title="Source Cluster ID",
                tickmode='array',
                tickvals=list(range(0, max(cluster_ids) + 50, 50)),
                ticktext=[str(i) for i in range(0, max(cluster_ids) + 50, 50)]
            ),
            yaxis_title="Number of Alignments",
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show some statistics
        non_zero_counts = [count for count in alignment_counts if count > 0]
        if non_zero_counts:
            st.write(f"**Average alignments per cluster:** {np.mean(alignment_counts):.2f}")
            st.write(f"**Maximum alignments:** {max(alignment_counts)}")
            st.write(f"**Minimum alignments:** {min(alignment_counts)}")
            st.write(f"**Clusters with no alignments:** {alignment_counts.count(0)}")
        else:
            st.warning("No alignments found above the selected threshold")
    
    with tab1:
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
        
        # Create dropdown for selecting encoder cluster
        encoder_clusters = sorted([align["encoder_id"] for align in alignment_data["alignments"]], 
                                key=lambda x: int(x.lstrip('c')))
        selected_encoder = st.selectbox(
            "Select Source Cluster",
            encoder_clusters,
            format_func=lambda x: f"Cluster {x.lstrip('c')}"
        )
        
        # Find the alignment data for selected encoder
        alignment = next(
            align for align in alignment_data["alignments"] 
            if align["encoder_id"] == selected_encoder
        )
        
        # Display total alignments count
        num_alignments = len(alignment['matches'])
        num_high_similarity = len([m for m in alignment['matches'] if m['similarity'] >= similarity_threshold])
        st.write(f"**Total alignments for Source Cluster {alignment['encoder_id'].lstrip('c')}:** {num_alignments}")
        st.write(f"**High similarity alignments (≥{similarity_threshold:.0%}):** {num_high_similarity}")
        st.info(f"Only showing alignments with similarity score of {similarity_threshold:.0%} or higher")
        
        # Filter and sort matches
        filtered_matches = [m for m in alignment['matches'] if m['similarity'] >= similarity_threshold]
        sorted_matches = sorted(filtered_matches, key=lambda x: int(x['decoder_id'].lstrip('c')))
        
        if filtered_matches:
            # Create dropdown for target clusters
            match_options = [
                f"Cluster {m['decoder_id'].lstrip('c')} (Similarity: {m['similarity']:.2%})"
                for m in sorted_matches
            ]
            selected_target = st.selectbox(
                "Select Target Cluster",
                range(len(match_options)),
                format_func=lambda i: match_options[i]
            )
            
            # Get selected match
            match = sorted_matches[selected_target]
            decoder_id = match['decoder_id']
            similarity = match['similarity']
            decoder_cluster = match['decoder_cluster']
            
            # Display clusters side by side
            st.write("#### Cluster Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("##### Source Cluster")
                with st.expander("Source Details", expanded=True):
                    st.write(f"**Cluster ID:** {alignment['encoder_id'].lstrip('c')}")
                    encoder_cluster = alignment['encoder_cluster']
                    st.write("**Unique Tokens:** " + ", ".join(encoder_cluster['Unique tokens']))
                    st.write(f"**Syntactic Label:** {encoder_cluster['Syntactic Label']}")
                    st.write("**Semantic Tags:**")
                    for tag in encoder_cluster['Semantic Tags']:
                        st.write(f"- {tag}")
                    st.write(f"**Description:** {encoder_cluster.get('Description', 'N/A')}")
                    
                    if encoder_sentences.get(f"c{selected_encoder.lstrip('c')}"):
                        st.write("**Context Sentences:**")
                        for sent_info in encoder_sentences[f"c{selected_encoder.lstrip('c')}"]:
                            tokens = sent_info["sentence"].split()
                            html = create_sentence_html(tokens, sent_info)
                            st.markdown(html, unsafe_allow_html=True)
            
            with col2:
                st.write(f"##### Target Cluster (Similarity: {similarity:.2%})")
                with st.expander("Target Details", expanded=True):
                    st.write(f"**Cluster ID:** {decoder_id.lstrip('c')}")
                    st.write("**Unique Tokens:** " + ", ".join(decoder_cluster['Unique tokens']))
                    st.write(f"**Syntactic Label:** {decoder_cluster['Syntactic Label']}")
                    st.write("**Semantic Tags:**")
                    for tag in decoder_cluster['Semantic Tags']:
                        st.write(f"- {tag}")
                    st.write(f"**Description:** {decoder_cluster.get('Description', 'N/A')}")
                    
                    if decoder_sentences.get(f"c{decoder_id.lstrip('c')}"):
                        st.write("**Context Sentences:**")
                        for sent_info in decoder_sentences[f"c{decoder_id.lstrip('c')}"]:
                            tokens = sent_info["sentence"].split()
                            html = create_sentence_html(tokens, sent_info)
                            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info(f"No aligned target clusters found with similarity ≥{similarity_threshold:.0%}")

def load_layer_alignment_metrics(model_base: str, selected_pair: str, layers: List[int]) -> dict:
    """Load and aggregate alignment metrics for all layers"""
    layer_metrics = {}
    
    for layer in layers:
        metrics_file = os.path.join(
            model_base,
            selected_pair,
            f"layer{layer}",
            "cluster_alignments.json"
        )
        
        print(f"Looking for metrics file: {metrics_file}")  # Debug print
        
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    
                # Calculate average scores for this layer
                calign_scores = []
                colap_scores = []
                
                for cluster_id, cluster_data in metrics.items():
                    metrics_data = cluster_data.get("metrics", {})
                    if "calign_score" in metrics_data:
                        calign_scores.append(float(metrics_data["calign_score"]))
                    if "colap_score" in metrics_data:
                        colap_scores.append(float(metrics_data["colap_score"]))
                
                if calign_scores or colap_scores:
                    layer_metrics[layer] = {
                        "avg_calign": sum(calign_scores) / len(calign_scores) if calign_scores else 0,
                        "avg_colap": sum(colap_scores) / len(colap_scores) if colap_scores else 0,
                        "num_clusters": len(calign_scores)  # Add count for debugging
                    }
                    print(f"Layer {layer}: Found {len(calign_scores)} clusters with metrics")  # Debug print
                else:
                    print(f"Layer {layer}: No valid scores found")  # Debug print
            except Exception as e:
                print(f"Error loading metrics for layer {layer}: {str(e)}")  # Debug print
                continue
        else:
            print(f"Metrics file not found for layer {layer}")  # Debug print
    
    return layer_metrics

def display_layer_alignment_metrics(model_base: str, selected_pair: str, layers: List[int]):
    """Display graphs for inter-layer alignment metrics"""
    print(f"Displaying metrics for layers: {layers}")  # Debug print
    
    # If only one layer is passed, get all available layers
    if len(layers) == 1:
        layers = get_available_layers(model_base, selected_pair)
        print(f"Expanded to all layers: {layers}")  # Debug print
    
    metrics = load_layer_alignment_metrics(model_base, selected_pair, layers)
    
    if not metrics:
        st.error("No alignment metrics found")
        print("No metrics found for any layer")  # Debug print
        return
    
    # Create lists for plotting
    layers = sorted(metrics.keys())
    avg_calign = [metrics[layer]["avg_calign"] for layer in layers]
    avg_colap = [metrics[layer]["avg_colap"] for layer in layers]
    
    print(f"Plotting metrics for layers: {layers}")  # Debug print
    print(f"Calign scores: {avg_calign}")  # Debug print
    print(f"Colap scores: {avg_colap}")  # Debug print
    
    # Add metrics table
    st.write("### Layer-wise Alignment Metrics")
    metrics_df = pd.DataFrame({
        'Layer': layers,
        'Avg Cluster Alignment': [f"{score:.2%}" for score in avg_calign],
        'Avg Cluster Overlap': [f"{score:.2%}" for score in avg_colap],
        'Clusters': [metrics[layer]["num_clusters"] for layer in layers]
    })
    st.dataframe(metrics_df)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=layers,
        y=[score * 100 for score in avg_calign],  # Convert to percentage
        name='Average Cluster Alignment Score',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=layers,
        y=[score * 100 for score in avg_colap],  # Convert to percentage
        name='Average Cluster Overlap Score',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Layer-wise Average Alignment Metrics - {model_base}',  # Changed back to model_base
            font=dict(weight='bold', size=20),
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title=dict(
            text='Layer',
            font=dict(weight='bold', size=14)
        ),
        yaxis_title=dict(
            text='Score (%)',
            font=dict(weight='bold', size=14)
        ),
        yaxis=dict(
            tickformat='.1f',
            range=[0, 110]  # Increased from 100 to 110 to show all points
        ),
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.01,
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

def analyze_keyword_evolution_standard(model_base: str, selected_pair: str, available_layers: List[int], keyword: str):
    """Analyze how a specific keyword evolves across layers in standard (non-mixed) models"""
    
    # Data structure to store analysis results for both encoder and decoder
    evolution_data = {
        'encoder': {
            'layers': [],
            'cluster_counts': [],
            'token_counts': [],
            'cluster_details': {}
        },
        'decoder': {
            'layers': [],
            'cluster_counts': [],
            'token_counts': [],
            'cluster_details': {}
        }
    }
    
    # Analyze each layer
    for layer in available_layers:
        # Analyze encoder clusters
        encoder_results = find_clusters_for_token_standard(
            model_base,
            selected_pair,
            layer,
            keyword,
            "encoder"
        )
        
        # Count encoder clusters and token occurrences
        encoder_clusters = 0
        encoder_tokens = 0
        encoder_cluster_info = {}
        
        for cluster_id, tokens in encoder_results.items():
            if keyword in tokens['matching_tokens']:
                encoder_clusters += 1
                token_count = sum(1 for t in tokens['all_tokens'] if t == keyword)
                encoder_tokens += token_count
                
                encoder_cluster_info[cluster_id] = {
                    'token_count': token_count,
                    'cluster_size': len(tokens['all_tokens']),
                    'token_percentage': token_count / len(tokens['all_tokens']) * 100
                }
        
        # Store encoder data
        evolution_data['encoder']['layers'].append(layer)
        evolution_data['encoder']['cluster_counts'].append(encoder_clusters)
        evolution_data['encoder']['token_counts'].append(encoder_tokens)
        evolution_data['encoder']['cluster_details'][layer] = encoder_cluster_info
        
        # Analyze decoder clusters
        decoder_results = find_clusters_for_token_standard(
            model_base,
            selected_pair,
            layer,
            keyword,
            "decoder"
        )
        
        # Count decoder clusters and token occurrences
        decoder_clusters = 0
        decoder_tokens = 0
        decoder_cluster_info = {}
        
        for cluster_id, tokens in decoder_results.items():
            if keyword in tokens['matching_tokens']:
                decoder_clusters += 1
                token_count = sum(1 for t in tokens['all_tokens'] if t == keyword)
                decoder_tokens += token_count
                
                decoder_cluster_info[cluster_id] = {
                    'token_count': token_count,
                    'cluster_size': len(tokens['all_tokens']),
                    'token_percentage': token_count / len(tokens['all_tokens']) * 100
                }
        
        # Store decoder data
        evolution_data['decoder']['layers'].append(layer)
        evolution_data['decoder']['cluster_counts'].append(decoder_clusters)
        evolution_data['decoder']['token_counts'].append(decoder_tokens)
        evolution_data['decoder']['cluster_details'][layer] = decoder_cluster_info
    
    return evolution_data

def display_keyword_evolution_standard(evolution_data: dict, keyword: str):
    """Display visualizations and analysis of keyword evolution for standard models"""
    st.write(f"### Evolution Analysis for '{keyword}'")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Combined View", "Detailed Analysis"])
    
    with tab1:
        # Create combined evolution graph
        fig = go.Figure()
        
        # Add encoder traces
        fig.add_trace(go.Scatter(
            x=evolution_data['encoder']['layers'],
            y=evolution_data['encoder']['cluster_counts'],
            name='Source Clusters',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=evolution_data['encoder']['layers'],
            y=evolution_data['encoder']['token_counts'],
            name='Source Token Occurrences',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2, dash='dot'),
            marker=dict(size=8)
        ))
        
        # Add decoder traces
        fig.add_trace(go.Scatter(
            x=evolution_data['decoder']['layers'],
            y=evolution_data['decoder']['cluster_counts'],
            name='Target Clusters',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=evolution_data['decoder']['layers'],
            y=evolution_data['decoder']['token_counts'],
            name='Target Token Occurrences',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Evolution of '{keyword}' Across Layers",
            xaxis_title="Layer",
            yaxis_title="Count",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create subtabs for encoder and decoder analysis
        subtab1, subtab2 = st.tabs(["Source (Encoder)", "Target (Decoder)"])
        
        with subtab1:
            st.write("#### Source Distribution")
            
            # Create heatmap for encoder clusters
            heatmap_data = []
            max_clusters = max(len(details) for details in evolution_data['encoder']['cluster_details'].values())
            
            for layer in evolution_data['encoder']['layers']:
                layer_data = evolution_data['encoder']['cluster_details'][layer]
                row = []
                for cluster_id in sorted(layer_data.keys(), key=lambda x: int(x[1:])):
                    row.append(layer_data[cluster_id]['token_percentage'])
                # Pad with zeros if needed
                row.extend([0] * (max_clusters - len(row)))
                heatmap_data.append(row)
            
            fig_encoder = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                y=evolution_data['encoder']['layers'],
                x=[f'Cluster {i+1}' for i in range(max_clusters)],
                colorscale='Viridis',
                colorbar=dict(title='Token %')
            ))
            
            fig_encoder.update_layout(
                title=f"Distribution of '{keyword}' Across Source Clusters and Layers",
                xaxis_title="Clusters",
                yaxis_title="Layer",
                height=400
            )
            
            st.plotly_chart(fig_encoder, use_container_width=True)
            
            # Display statistics
            st.write("**Statistics:**")
            encoder_stats = pd.DataFrame({
                'Layer': evolution_data['encoder']['layers'],
                'Clusters with Token': evolution_data['encoder']['cluster_counts'],
                'Total Occurrences': evolution_data['encoder']['token_counts']
            })
            st.dataframe(encoder_stats)
        
        with subtab2:
            st.write("#### Target Distribution")
            
            # Create heatmap for decoder clusters
            heatmap_data = []
            max_clusters = max(len(details) for details in evolution_data['decoder']['cluster_details'].values())
            
            for layer in evolution_data['decoder']['layers']:
                layer_data = evolution_data['decoder']['cluster_details'][layer]
                row = []
                for cluster_id in sorted(layer_data.keys(), key=lambda x: int(x[1:])):
                    row.append(layer_data[cluster_id]['token_percentage'])
                # Pad with zeros if needed
                row.extend([0] * (max_clusters - len(row)))
                heatmap_data.append(row)
            
            fig_decoder = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                y=evolution_data['decoder']['layers'],
                x=[f'Cluster {i+1}' for i in range(max_clusters)],
                colorscale='Viridis',
                colorbar=dict(title='Token %')
            ))
            
            fig_decoder.update_layout(
                title=f"Distribution of '{keyword}' Across Target Clusters and Layers",
                xaxis_title="Clusters",
                yaxis_title="Layer",
                height=400
            )
            
            st.plotly_chart(fig_decoder, use_container_width=True)
            
            # Display statistics
            st.write("**Statistics:**")
            decoder_stats = pd.DataFrame({
                'Layer': evolution_data['decoder']['layers'],
                'Clusters with Token': evolution_data['decoder']['cluster_counts'],
                'Total Occurrences': evolution_data['decoder']['token_counts']
            })
            st.dataframe(decoder_stats)

def display_individual_cluster_labels(cluster_id: str, model_base: str, selected_pair: str, layer: int, component: str):
    """Display Gemini labels for an individual cluster"""
    
    # Add debug prints
    labels_file = Path(model_base) / selected_pair / f"layer{layer}" / f"{component}_gemini_labels.json"
    
    if not labels_file.exists():
        st.warning("Gemini labels not found for this cluster")
        return
        
    with open(labels_file) as f:
        labels_data = json.load(f)
    
    # Find labels for this cluster
    cluster_labels = next((item[cluster_id] for item in labels_data if cluster_id in item), None)
    
    if cluster_labels:
        # Remove the expander and just show the content directly
        st.write("### Gemini Analysis")
        st.write("**Syntactic Label:**", cluster_labels.get("Syntactic Label", "N/A"))
        
        st.write("**Semantic Tags:**")
        for tag in cluster_labels.get("Semantic Tags", []):
            st.write(f"- {tag}")
            
        st.write("**Description:**", cluster_labels.get("Description", "N/A"))
    else:
        print(f"No labels found for cluster {cluster_id}")
        st.warning("No Gemini labels found for this cluster")

def display_cluster_details(cluster_data, cluster_id, model_base, selected_pair, layer, component):
    """Display detailed information for a single cluster"""
    
    # Display existing cluster information
    st.write(f"### Cluster {cluster_id}")
    
    # Display tokens
    st.write("**Unique Tokens:**")
    st.write(", ".join(cluster_data["Unique tokens"]))
    
    # Add Gemini labels
    display_individual_cluster_labels(
        f"c{cluster_id}", 
        model_base,
        selected_pair,
        layer,
        component
    )
    
    # Display context sentences if available
    if "Context Sentences" in cluster_data:
        st.write("**Context Sentences:**")
        for sentence in cluster_data["Context Sentences"]:
            st.write(f"- {sentence}")

def display_mixed_cluster_labels(cluster_id: str, model_base: str, selected_pair: str, layer: int):
    """Display Gemini labels for mixed model clusters"""
    
    labels_file = Path(model_base) / selected_pair / f"layer{layer}" / "mixed_gemini_labels.json"
    
    if not labels_file.exists():
        st.warning("Mixed model Gemini labels not found for this cluster")
        return
        
    with open(labels_file) as f:
        labels_data = json.load(f)
    
    cluster_labels = next((item[cluster_id] for item in labels_data if cluster_id in item), None)
    
    if cluster_labels:
        with st.expander("Cross-Language Analysis", expanded=True):
            st.write("**Lexical Patterns:**", cluster_labels.get("lexical_patterns", "N/A"))
            
            st.write("**Semantic Tags:**")
            for tag in cluster_labels.get("semantic_tags", []):
                st.write(f"- {tag}")
            
            st.write("**Functional Equivalence:**", cluster_labels.get("functional_equivalence", "N/A"))
            st.write("**Semantic Description:**", cluster_labels.get("semantic_description", "N/A"))
    else:
        st.warning("No mixed model labels found for this cluster")

def find_clusters_with_multiple_tokens(model_base: str, selected_pair: str, selected_layer: int, tokens: List[str], component: str):
    """Find clusters containing all specified tokens"""
    cluster_file = os.path.join(
        model_base, 
        selected_pair,
        f"layer{selected_layer}",
        f"{component}-clusters-kmeans-500.txt"
    )
    
    # Dictionary to store clusters containing the tokens
    token_clusters = {}
    
    # Check if file exists before trying to open it
    if not os.path.exists(cluster_file):
        print(f"Warning: Cluster file not found: {cluster_file}")
        return token_clusters  # Return empty dictionary
    
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|||')
            if len(parts) == 5:  # token|||other|||sent_id|||token_idx|||cluster_id
                token = parts[0].strip()
                cluster_id = f"c{parts[4].strip()}"
                
                # Initialize cluster entry if not exists
                if cluster_id not in token_clusters:
                    token_clusters[cluster_id] = {
                        'matching_tokens': {t: set() for t in tokens},
                        'all_tokens': set()
                    }
                
                # Check if token matches any of our search tokens
                for search_token in tokens:
                    if search_token.lower() in token.lower():
                        token_clusters[cluster_id]['matching_tokens'][search_token].add(token)
                
                token_clusters[cluster_id]['all_tokens'].add(token)
    
    # Filter clusters to only include those containing all search tokens
    filtered_clusters = {
        cluster_id: data for cluster_id, data in token_clusters.items()
        if all(len(matches) > 0 for matches in data['matching_tokens'].values())
    }
    
    return filtered_clusters

def analyze_token_evolution(model_base: str, selected_pair: str, available_layers: List[int], tokens: List[str]):
    """Analyze how tokens evolve across layers"""
    evolution_data = {
        'layers': available_layers,
        'individual_counts': {token: [] for token in tokens},
        'combined_counts': [] if len(tokens) > 1 else None
    }
    
    # Check if it's a mixed model
    is_mixed = os.path.exists(os.path.join(model_base, selected_pair, f"layer{available_layers[0]}", "clusters-kmeans-500.txt"))
    
    for layer in available_layers:
        if is_mixed:
            # For mixed models
            cluster_file = os.path.join(model_base, selected_pair, f"layer{layer}", "clusters-kmeans-500.txt")
            token1_clusters = set()
            token2_clusters = set()
            
            if os.path.exists(cluster_file):
                with open(cluster_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('|||')
                        if len(parts) >= 5:
                            token = parts[0].strip()
                            cluster_id = f"c{parts[4].strip()}"
                            
                            if tokens[0].lower() in token.lower():
                                token1_clusters.add(cluster_id)
                            if len(tokens) > 1 and tokens[1].lower() in token.lower():
                                token2_clusters.add(cluster_id)
            
            evolution_data['individual_counts'][tokens[0]].append(len(token1_clusters))
            
            if len(tokens) > 1:
                evolution_data['individual_counts'][tokens[1]].append(len(token2_clusters))
                evolution_data['combined_counts'].append(len(token1_clusters.intersection(token2_clusters)))
        
        else:
            # For standard models
            token1_clusters_encoder = set()
            token1_clusters_decoder = set()
            token2_clusters_encoder = set()
            token2_clusters_decoder = set()
            
            # Check encoder clusters
            encoder_file = os.path.join(model_base, selected_pair, f"layer{layer}", "encoder-clusters-kmeans-500.txt")
            if os.path.exists(encoder_file):
                with open(encoder_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('|||')
                        if len(parts) >= 5:
                            token = parts[0].strip()
                            cluster_id = f"c{parts[4].strip()}"
                            
                            if tokens[0].lower() in token.lower():
                                token1_clusters_encoder.add(cluster_id)
                            if len(tokens) > 1 and tokens[1].lower() in token.lower():
                                token2_clusters_encoder.add(cluster_id)
            
            # Check decoder clusters
            decoder_file = os.path.join(model_base, selected_pair, f"layer{layer}", "decoder-clusters-kmeans-500.txt")
            if os.path.exists(decoder_file):
                with open(decoder_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('|||')
                        if len(parts) >= 5:
                            token = parts[0].strip()
                            cluster_id = f"c{parts[4].strip()}"
                            
                            if tokens[0].lower() in token.lower():
                                token1_clusters_decoder.add(cluster_id)
                            if len(tokens) > 1 and tokens[1].lower() in token.lower():
                                token2_clusters_decoder.add(cluster_id)
            
            # Combine encoder and decoder results
            token1_clusters = token1_clusters_encoder.union(token1_clusters_decoder)
            evolution_data['individual_counts'][tokens[0]].append(len(token1_clusters))
            
            if len(tokens) > 1:
                token2_clusters = token2_clusters_encoder.union(token2_clusters_decoder)
                evolution_data['individual_counts'][tokens[1]].append(len(token2_clusters))
                evolution_data['combined_counts'].append(len(token1_clusters.intersection(token2_clusters)))
    
    # Add debug information
    st.write(f"Debug: Evolution data collected for layers {evolution_data['layers']}")
    st.write(f"Debug: Token counts: {evolution_data['individual_counts']}")
    if len(tokens) > 1:
        st.write(f"Debug: Combined counts: {evolution_data['combined_counts']}")
    
    return evolution_data

def display_token_evolution(evolution_data: dict, tokens: List[str]):
    """Display token evolution analysis"""
    if not evolution_data or not tokens:
        return

    fig = go.Figure()

    # Add combined occurrences
    if len(tokens) > 1 and evolution_data['combined_counts']:
        fig.add_trace(go.Scatter(
            x=evolution_data['layers'],
            y=evolution_data['combined_counts'],
            name='Co-occurring',
            mode='lines+markers',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=8)
        ))

    # Add individual token traces
    colors = ['#3498db', '#e74c3c']  # Blue and Red for individual tokens
    for i, token in enumerate(tokens):
        # Add total occurrences
        fig.add_trace(go.Scatter(
            x=evolution_data['layers'],
            y=evolution_data['individual_counts'][token],
            name=f"Total '{token}'",
            mode='lines+markers',
            line=dict(color=colors[i], width=2),
            marker=dict(size=8)
        ))

        if len(tokens) > 1 and evolution_data['combined_counts']:
            # Calculate exclusive occurrences (total minus co-occurrences)
            exclusive_counts = [
                total - combined 
                for total, combined in zip(
                    evolution_data['individual_counts'][token],
                    evolution_data['combined_counts']
                )
            ]
            
            # Add exclusive occurrences
            fig.add_trace(go.Scatter(
                x=evolution_data['layers'],
                y=exclusive_counts,
                name=f"Exclusive '{token}'",
                mode='lines+markers',
                line=dict(color=colors[i], dash='dot', width=2),
                marker=dict(size=8)
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text='Token Evolution Analysis',
            font=dict(size=20)
        ),
        xaxis_title=dict(
            text='Layer',
            font=dict(size=14)
        ),
        yaxis_title=dict(
            text='Number of Clusters',
            font=dict(size=14)
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Add gridlines
    fig.update_xaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
    fig.update_yaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')

    st.plotly_chart(fig, use_container_width=True)

def get_predefined_token_pairs():
    """Return predefined token pairs organized by categories"""
    return {
        "Control Flow": {
            "description": "Different control flow constructs",
            "pairs": [
                ("for", "while"),
                ("if", "switch"),
                ("break", "continue"),
                ("try", "catch")
            ]
        },
        "Access Modifiers": {
            "description": "Access and modifier keywords",
            "pairs": [
                ("public", "private"),
                ("static", "final"),
                ("abstract", "interface")
            ]
        },
        "Variable/Type": {
            "description": "Variable and type-related tokens",
            "pairs": [
                ("int", "Integer"),
                ("null", "Optional"),
                ("var", "String")  # Example of var vs explicit type
            ]
        },
        "Collections": {
            "description": "Collection-related tokens",
            "pairs": [
                ("List", "Array"),
                ("ArrayList", "LinkedList"),
                ("HashMap", "TreeMap"),
                ("Set", "List")
            ]
        },
        "Threading": {
            "description": "Threading and concurrency tokens",
            "pairs": [
                ("synchronized", "volatile"),
                ("Runnable", "Callable"),
                ("wait", "sleep")
            ]
        },
        "Object-Oriented": {
            "description": "Object-oriented programming tokens",
            "pairs": [
                ("extends", "implements"),
                ("this", "super"),
                ("new", "clone")
            ]
        }
    }

def display_token_pair_analysis(model_name, model_base, selected_pair, available_layers):
    """Display analysis for predefined token pairs"""
    st.write("## Token Pair Analysis")
    
    # Get predefined token pairs
    token_pairs = get_predefined_token_pairs()
    
    # Create tabs for each category
    tabs = st.tabs(list(token_pairs.keys()))
    
    # Determine if we're using a mixed model
    is_mixed = os.path.exists(os.path.join(model_base, selected_pair, f"layer{available_layers[0]}", "clusters-kmeans-500.txt"))
    
    # Determine available components
    available_components = []
    if is_mixed:
        available_components = ["mixed"]
    else:
        # Check which components exist
        if os.path.exists(os.path.join(model_base, selected_pair, f"layer{available_layers[0]}", "encoder-clusters-kmeans-500.txt")):
            available_components.append("encoder")
        if os.path.exists(os.path.join(model_base, selected_pair, f"layer{available_layers[0]}", "decoder-clusters-kmeans-500.txt")):
            available_components.append("decoder")
    
    for tab, (category, data) in zip(tabs, token_pairs.items()):
        with tab:
            st.write(f"### {category}")
            st.write(data["description"])
            
            # Create a section for each pair in the category
            for token1, token2 in data["pairs"]:
                with st.expander(f"{token1} vs {token2}"):
                    # Initialize evolution data for both tokens if not in session state
                    pair_key = f"{token1}_{token2}_evolution"
                    if pair_key not in st.session_state:
                        with st.spinner(f"Analyzing evolution of '{token1}' and '{token2}'..."):
                            evolution_data = analyze_token_evolution(
                                model_base,
                                selected_pair,
                                available_layers,
                                [token1, token2]
                            )
                            st.session_state[pair_key] = evolution_data
                    
                    # Display evolution analysis
                    if st.session_state[pair_key]:
                        display_token_evolution(
                            st.session_state[pair_key],
                            [token1, token2]
                        )
                    
                    # Add cluster analysis using existing functionality
                    st.write("#### Cluster Analysis")
                    
                    for component in available_components:
                        st.write(f"**{component.title()} Component Analysis**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**{token1} Clusters**")
                            if is_mixed:
                                clusters = find_clusters_for_token(
                                    model_base,
                                    selected_pair,
                                    available_layers[0],
                                    token1
                                )
                            else:
                                clusters = find_clusters_for_token_standard(
                                    model_base,
                                    selected_pair,
                                    available_layers[0],
                                    token1,
                                    component
                                )
                            if clusters:
                                for cluster_id, tokens in clusters.items():
                                    st.write(f"Cluster {cluster_id}: {', '.join(tokens)}")
                            else:
                                st.write(f"No clusters found for '{token1}'")
                        
                        with col2:
                            st.write(f"**{token2} Clusters**")
                            if is_mixed:
                                clusters = find_clusters_for_token(
                                    model_base,
                                    selected_pair,
                                    available_layers[0],
                                    token2
                                )
                            else:
                                clusters = find_clusters_for_token_standard(
                                    model_base,
                                    selected_pair,
                                    available_layers[0],
                                    token2,
                                    component
                                )
                            if clusters:
                                for cluster_id, tokens in clusters.items():
                                    st.write(f"Cluster {cluster_id}: {', '.join(tokens)}")
                            else:
                                st.write(f"No clusters found for '{token2}'")
                        
                        # Add co-occurrence analysis
                        st.write("#### Co-occurrence Analysis")
                        cooccurring_clusters = find_clusters_with_multiple_tokens(
                            model_base,
                            selected_pair,
                            available_layers[0],
                            [token1, token2],
                            component
                        )
                        
                        if cooccurring_clusters:
                            st.write(f"Found {len(cooccurring_clusters)} clusters with both tokens")
                            for cluster_id, data in cooccurring_clusters.items():
                                # Use a container instead of an expander
                                container = st.container()
                                container.write(f"**Cluster {cluster_id}**")
                                container.write(f"**{token1} occurrences:** {', '.join(data['matching_tokens'][token1])}")
                                container.write(f"**{token2} occurrences:** {', '.join(data['matching_tokens'][token2])}")
                                container.write("**All tokens in cluster:**")
                                container.write(", ".join(sorted(data['all_tokens'])))
                                container.write("---")  # Add a separator between clusters
                        else:
                            st.write("No clusters found containing both tokens")

def find_clusters_by_semantic_tag(model_base: str, selected_pair: str, layer: int, search_tag: str, component: str) -> dict:
    """Find clusters that have the specified semantic tag in their Gemini labels"""
    labels_file = Path(model_base) / selected_pair / f"layer{layer}" / f"{component}_gemini_labels.json"
    cluster_file = os.path.join(
        model_base, 
        selected_pair,
        f"layer{layer}",
        f"{component}-clusters-kmeans-500.txt"
    )
    
    if not labels_file.exists() or not os.path.exists(cluster_file):
        return {}
        
    matching_clusters = {}
    cluster_tokens = defaultdict(set)  # Use defaultdict for easier token collection
    
    # First load all tokens for each cluster
    try:
        with open(cluster_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|||')
                if len(parts) == 5:  # token|||other|||sent_id|||token_idx|||cluster_id
                    token = parts[0].strip()
                    cluster_id = f"c{parts[4].strip()}"
                    cluster_tokens[cluster_id].add(token)
    except Exception as e:
        print(f"Error reading cluster file: {e}")
        return {}
    
    # Then load and match semantic tags
    try:
        with open(labels_file) as f:
            labels_data = json.load(f)
        
        # Search through all clusters
        for cluster_labels in labels_data:
            for cluster_id, labels in cluster_labels.items():
                # Check if any semantic tag matches the search term (case-insensitive)
                if any(search_tag.lower() in tag.lower() for tag in labels.get("Semantic Tags", [])):
                    matching_clusters[cluster_id] = {
                        'semantic_tags': labels.get("Semantic Tags", []),
                        'syntactic_label': labels.get("Syntactic Label", "N/A"),
                        'description': labels.get("Description", "N/A"),
                        'tokens': sorted(cluster_tokens.get(cluster_id, set()))  # Ensure default empty set
                    }
    except Exception as e:
        print(f"Error processing labels: {e}")
        return {}
    
    # Debug print
    print(f"Found {len(matching_clusters)} matching clusters")
    for cluster_id, data in matching_clusters.items():
        print(f"Cluster {cluster_id}: {len(data.get('tokens', []))} tokens")
    
    return matching_clusters

def analyze_semantic_tag_evolution(model_base: str, selected_pair: str, available_layers: List[int], search_tag: str):
    """Analyze how a semantic tag evolves across layers"""
    evolution_data = {
        'layers': available_layers,
        'encoder_counts': [],
        'decoder_counts': [],
        'cluster_details': {
            'encoder': {},
            'decoder': {}
        }
    }
    
    for layer in available_layers:
        # Get encoder clusters with this semantic tag
        encoder_clusters = find_clusters_by_semantic_tag(
            model_base,
            selected_pair,
            layer,
            search_tag,
            "encoder"
        )
        
        # Get decoder clusters with this semantic tag
        decoder_clusters = find_clusters_by_semantic_tag(
            model_base,
            selected_pair,
            layer,
            search_tag,
            "decoder"
        )
        
        # Store counts
        evolution_data['encoder_counts'].append(len(encoder_clusters))
        evolution_data['decoder_counts'].append(len(decoder_clusters))
        
        # Store detailed information
        evolution_data['cluster_details']['encoder'][layer] = encoder_clusters
        evolution_data['cluster_details']['decoder'][layer] = decoder_clusters
    
    return evolution_data

def display_semantic_tag_evolution(evolution_data: dict, search_tag: str):
    """Display evolution analysis for semantic tags"""
    st.write(f"### Evolution Analysis for Semantic Tag: '{search_tag}'")
    
    # Create main evolution graph
    fig = go.Figure()
    
    # Add encoder trace
    fig.add_trace(go.Scatter(
        x=evolution_data['layers'],
        y=evolution_data['encoder_counts'],
        name='Source Clusters',
        mode='lines+markers',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8)
    ))
    
    # Add decoder trace
    fig.add_trace(go.Scatter(
        x=evolution_data['layers'],
        y=evolution_data['decoder_counts'],
        name='Target Clusters',
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Evolution of Semantic Tag "{search_tag}" Across Layers',
            font=dict(size=20)
        ),
        xaxis_title=dict(
            text='Layer',
            font=dict(size=14)
        ),
        yaxis_title=dict(
            text='Number of Clusters',
            font=dict(size=14)
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add gridlines
    fig.update_xaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
    fig.update_yaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics table
    st.write("### Detailed Statistics")
    stats_data = {
        'Layer': evolution_data['layers'],
        'Source Clusters': evolution_data['encoder_counts'],
        'Target Clusters': evolution_data['decoder_counts'],
        'Total Clusters': [e + d for e, d in zip(evolution_data['encoder_counts'], evolution_data['decoder_counts'])]
    }
    
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats)
    
    # Add detailed cluster information in expandable sections
    st.write("### Cluster Details by Layer")
    tab1, tab2 = st.tabs(["Source Clusters", "Target Clusters"])
    
    with tab1:
        for layer in evolution_data['layers']:
            with st.expander(f"Layer {layer}"):
                clusters = evolution_data['cluster_details']['encoder'][layer]
                if clusters:
                    for cluster_id, data in clusters.items():
                        st.write(f"**Cluster {cluster_id}**")
                        st.write("Semantic Tags:", ", ".join(data['semantic_tags']))
                        st.write("Syntactic Label:", data['syntactic_label'])
                        st.write("Description:", data['description'])
                        st.write("Tokens:", ", ".join(data['tokens'][:10]) + ("..." if len(data['tokens']) > 10 else ""))
                        st.write("---")
                else:
                    st.write("No clusters found in this layer")
    
    with tab2:
        for layer in evolution_data['layers']:
            with st.expander(f"Layer {layer}"):
                clusters = evolution_data['cluster_details']['decoder'][layer]
                if clusters:
                    for cluster_id, data in clusters.items():
                        st.write(f"**Cluster {cluster_id}**")
                        st.write("Semantic Tags:", ", ".join(data['semantic_tags']))
                        st.write("Syntactic Label:", data['syntactic_label'])
                        st.write("Description:", data['description'])
                        st.write("Tokens:", ", ".join(data['tokens'][:10]) + ("..." if len(data['tokens']) > 10 else ""))
                        st.write("---")
                else:
                    st.write("No clusters found in this layer")

def handle_semantic_tag_search(model_name, model_base, selected_pair, available_layers):
    """Handle semantic tag search functionality"""
    st.write("### Semantic Tag Search")
    
    # Initialize session state for semantic search if not exists
    if 'semantic_search_state' not in st.session_state:
        st.session_state.semantic_search_state = {
            'matching_tags': [],
            'last_search': None,
            'last_analyzed_tag': None,
            'encoder_clusters': {},
            'decoder_clusters': {}
        }
    
    # Search box for semantic tags
    search_term = st.text_input(
        "Search for semantic tags:",
        help="Type to search for semantic tags (e.g., 'loop', 'condition', 'memory')"
    )
    
    if search_term and search_term != st.session_state.semantic_search_state['last_search']:
        with st.spinner("Searching for matching semantic tags..."):
            matching_tags = find_matching_semantic_tags(
                model_base,
                selected_pair,
                available_layers[0],
                search_term
            )
            
            st.session_state.semantic_search_state.update({
                'matching_tags': matching_tags,
                'last_search': search_term
            })
    
    matching_tags = st.session_state.semantic_search_state['matching_tags']
    
    if matching_tags:
        selected_tag = st.selectbox(
            "Select a semantic tag:",
            matching_tags,
            key="semantic_tag_selector"
        )
        
        if selected_tag and st.button("Analyze Semantic Tag", type="primary"):
            # Check if we need to recompute analysis
            if ('last_analyzed_tag' not in st.session_state.semantic_search_state or
                st.session_state.semantic_search_state['last_analyzed_tag'] != selected_tag):
                
                with st.spinner(f"Analyzing semantic tag '{selected_tag}'..."):
                    # Fix: Call find_clusters_by_semantic_tag separately for encoder and decoder
                    encoder_clusters = find_clusters_by_semantic_tag(
                        model_base,
                        selected_pair,
                        available_layers[0],
                        selected_tag,
                        "encoder"  # Add the missing component parameter
                    )
                    
                    decoder_clusters = find_clusters_by_semantic_tag(
                        model_base,
                        selected_pair,
                        available_layers[0],
                        selected_tag,
                        "decoder"  # Add the missing component parameter
                    )
                    
                    st.session_state.semantic_search_state.update({
                        'last_analyzed_tag': selected_tag,
                        'encoder_clusters': encoder_clusters,
                        'decoder_clusters': decoder_clusters
                    })
            
            # Display results
            encoder_clusters = st.session_state.semantic_search_state['encoder_clusters']
            decoder_clusters = st.session_state.semantic_search_state['decoder_clusters']
            
            if encoder_clusters or decoder_clusters:
                tab1, tab2 = st.tabs(["Source (Encoder) Clusters", "Target (Decoder) Clusters"])
                
                with tab1:
                    st.write(f"### Source Clusters with Tag: {selected_tag}")
                    if encoder_clusters:
                        for cluster_id, tokens in encoder_clusters.items():
                            with st.expander(f"Cluster {cluster_id}"):
                                st.write("**Tokens in Cluster:**", ", ".join(tokens))
                                display_individual_cluster_labels(
                                    cluster_id,
                                    model_base,
                                    selected_pair,
                                    available_layers[0],
                                    "encoder"
                                )
                    else:
                        st.info("No source clusters found with this semantic tag")
                
                with tab2:
                    st.write(f"### Target Clusters with Tag: {selected_tag}")
                    if decoder_clusters:
                        for cluster_id, tokens in decoder_clusters.items():
                            with st.expander(f"Cluster {cluster_id}"):
                                st.write("**Tokens in Cluster:**", ", ".join(tokens))
                                display_individual_cluster_labels(
                                    cluster_id,
                                    model_base,
                                    selected_pair,
                                    available_layers[0],
                                    "decoder"
                                )
                    else:
                        st.info("No target clusters found with this semantic tag")
            else:
                st.warning(f"No clusters found with semantic tag: {selected_tag}")
    elif search_term:
        st.info("No matching semantic tags found")

def display_filtered_clusters(filtered_results):
    """Helper function to display filtered cluster results"""
    for cluster_id, data in filtered_results.items():
        with st.expander(f"Cluster {cluster_id}"):
            st.write("**Semantic Tags:**")
            for tag in data['semantic_tags']:
                st.write(f"- {tag}")
            st.write(f"**Syntactic Label:** {data['syntactic_label']}")
            st.write(f"**Description:** {data['description']}")
            st.write("**Tokens:**", ", ".join(data['tokens']))

# def handle_mixed_model_view(model_name, model_base, selected_pair, selected_layer, available_layers):
#     """Handle view for mixed models"""
#     # Add tabs for different views
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "Cluster Browser",
#         "Language Distribution",
#         "Predefined Keywords",
#         "Token Search",
#         "Token Pairs"  # New tab
#     ])

#     with tab1:
#         display_mixed_clusters(model_name, model_base, selected_pair, selected_layer)
        
#     with tab2:
#         display_language_distribution(model_base, selected_pair, available_layers)
        
#     with tab3:
#         add_predefined_keywords_tab(model_name, model_base, selected_pair, available_layers)
        
#     with tab4:
#         handle_token_search(model_name, model_base, selected_pair, available_layers)
    
#     with tab5:
#         display_token_pair_analysis(model_name, model_base, selected_pair, available_layers)

def main():
    st.set_page_config(layout="wide", page_title="Code Concept Explorer")
    
    st.title("Code Concept Cluster Explorer")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["t5", "coderosetta_ft", "coderosetta_daebt", "coderosetta_aer", "coderosetta_mlm"], #"coderosetta_mlm_mixed", "coderosetta_aer_mixed"],
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


    handle_standard_model_view(model_name, model_base, selected_pair, selected_layer)

if __name__ == "__main__":
    main()