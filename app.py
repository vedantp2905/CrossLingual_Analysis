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
            # Get and display language statistics
            stats = get_language_statistics(sentences, os.path.join(model, model_pair.split('/')[1]))
            display_language_statistics(stats)
            
            # Remove this section that was showing additional context sentences
            # Only show sentences within the language statistics tabs
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
    
    for idx, token in enumerate(tokens):
        if idx == sent_info["token_idx"]:
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
    for cluster_id, sentences in cluster_sentences.items():
        stats = get_language_statistics(sentences, os.path.join(model_base, selected_pair))
        if stats and stats["mixed_count"] > 0:
            mixed_count += 1
            
    return mixed_count

def count_language_dominated_clusters(model_base: str, selected_pair: str, selected_layer: int) -> dict:
    """Count clusters dominated by each language"""
    stats = {
        "cpp_dominated": 0,
        "cuda_dominated": 0,
        "mixed": 0,  # For cases with equal tokens
        "total": 0
    }
    
    # Load cluster sentences
    cluster_sentences = load_cluster_sentences(
        os.path.join(model_base, selected_pair),
        selected_layer,
        "mixed"
    )
    
    # For each cluster, analyze language distribution
    for cluster_id, sentences in cluster_sentences.items():
        lang_stats = get_language_statistics(sentences, os.path.join(model_base, selected_pair))
        if lang_stats:
            cpp_count = lang_stats["cpp_count"]
            cuda_count = lang_stats["cuda_count"]
            
            # A cluster is dominated by whichever language has more tokens
            if cpp_count > cuda_count:
                stats["cpp_dominated"] += 1
            elif cuda_count > cpp_count:
                stats["cuda_dominated"] += 1
            else:  # Equal counts
                stats["mixed"] += 1
                
            stats["total"] += 1
            
    return stats

def display_layer_statistics(model_base: str, selected_pair: str, available_layers: List[int]):
    """Display language statistics for all layers"""
    st.write("### Layer-wise Language Distribution")
    
    # Create columns for metrics
    cols = st.columns(len(available_layers))
    
    # Get stats for each layer
    for idx, layer in enumerate(available_layers):
        stats = count_language_dominated_clusters(model_base, selected_pair, layer)
        
        with cols[idx]:
            st.write(f"**Layer {layer}**")
            total = stats["total"]
            if total > 0:
                st.metric("C++ Dominated", 
                    f"{stats['cpp_dominated']} ({(stats['cpp_dominated']/total)*100:.1f}%)")
                st.metric("CUDA Dominated", 
                    f"{stats['cuda_dominated']} ({(stats['cuda_dominated']/total)*100:.1f}%)")
                if stats["mixed"] > 0:  # Only show mixed if there are any
                    st.metric("Mixed (Equal)", 
                        f"{stats['mixed']} ({(stats['mixed']/total)*100:.1f}%)")
            else:
                st.write("No clusters found")

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

def main():
    st.set_page_config(layout="wide", page_title="Code Concept Explorer")
    
    st.title("Code Concept Cluster Explorer")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Update model selection dropdown to include coderosetta_aer_mixed
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["t5", "coderosetta", "coderosetta_aer", "coderosetta_mlm", "coderosetta_mlm_mixed", "coderosetta_aer_mixed"]
    )
    model_base = os.path.join(model_name)
    
    # Get available language pairs
    lang_pairs = [d for d in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, d))]
    if not lang_pairs:
        st.error("No language pairs found in the specified directory")
        return
        
    selected_pair = st.sidebar.selectbox("Language Pair", lang_pairs)
    
    # Update view selection to include both mixed models
    if model_name in ["coderosetta_mlm_mixed", "coderosetta_aer_mixed"]:
        view = "Individual Clusters"
    else:
        view = st.sidebar.radio(
            "View", 
            ["Individual Clusters", "Aligned Clusters", "Top Semantic Tags"]
        )
    
    # Initialize session state for cluster index if not exists
    if 'current_cluster_index' not in st.session_state:
        st.session_state.current_cluster_index = 0
    
    # Get available layers
    available_layers = get_available_layers(model_base, selected_pair)
    
    if not available_layers:
        st.error("No layers found with valid data")
        return
    
    # Get and validate selected layer
    selected_layer = st.sidebar.selectbox(
        "Layer",
        available_layers,
        format_func=lambda x: f"Layer {x}"
    )
    
    if selected_layer is None and available_layers:
        selected_layer = available_layers[0]
    
    if view == "Top Semantic Tags" and model_name not in ["coderosetta_mlm_mixed", "coderosetta_aer_mixed"]:
        display_top_semantic_tags(model_base, selected_pair)
    elif view == "Individual Clusters":
        if model_name in ["coderosetta_mlm_mixed", "coderosetta_aer_mixed"]:
            # Display dataset balance at the top
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
                st.markdown("---")  # Add a separator
            
            # Display layer-wise statistics
            display_layer_statistics(model_base, selected_pair, available_layers)
            st.markdown("---")  # Add a separator
            
            # Rest of the existing code for mixed clusters...
            mixed_clusters_count = count_mixed_clusters(model_base, selected_pair, selected_layer)
            st.write(f"### Layer {selected_layer} Statistics")
            st.metric("Clusters with Mixed Language Tokens", mixed_clusters_count)
            st.markdown("---")  # Add a separator
            
            # First load mixed cluster sentences
            cluster_sentences = load_cluster_sentences(
                os.path.join(model_base, selected_pair),
                selected_layer,
                "mixed"
            )
            
            # Get cluster IDs right after loading sentences
            cluster_ids = sorted(list(cluster_sentences.keys()))
            
            if not cluster_ids:
                st.error("No clusters found")
                return
                
            # Now add search functionality
            st.sidebar.write("### Token Search")
            search_token = st.sidebar.text_input("Search for token:", key="token_search")
            
            if search_token:
                token_clusters = find_clusters_for_token(model_base, selected_pair, selected_layer, search_token)
                
                if token_clusters:
                    st.sidebar.write(f"Found in {len(token_clusters)} clusters:")
                    cluster_options = []
                    for cluster_id, tokens in token_clusters.items():
                        if cluster_id in cluster_ids:  # Verify cluster exists in current view
                            token_list = ", ".join(tokens[:3])  # Show first 3 tokens
                            if len(tokens) > 3:
                                token_list += "..."
                            cluster_options.append(f"{cluster_id}: {token_list}")
                    
                    if cluster_options:
                        selected_search_result = st.sidebar.selectbox(
                            "Select cluster to view:",
                            cluster_options,
                            key="search_cluster_select"
                        )
                        
                        # Add a button to navigate to the selected cluster
                        if st.sidebar.button("Go to Cluster"):
                            cluster_id = selected_search_result.split(":")[0].strip()
                            st.session_state.current_cluster_index = cluster_ids.index(cluster_id)
                else:
                    st.sidebar.write("No clusters found containing this token.")
            
            st.header(f"Mixed Clusters - Layer {selected_layer}")
            
            selected_cluster = st.selectbox(
                "Select Cluster",
                cluster_ids,
                index=st.session_state.current_cluster_index,
                key="cluster_selector"
            )
            
            # Create minimal cluster data structure
            cluster_data = {"Unique tokens": []}  # Empty as we don't have token info
            context_sentences = cluster_sentences.get(selected_cluster, [])
            
            display_cluster_info(
                cluster_data,
                f"{model_name}/{selected_pair}",
                selected_layer,
                selected_cluster,
                sentences=context_sentences
            )
        else:
            component = st.sidebar.radio("Component", ["encoder", "decoder"])
            # Load labels
            labels_file = os.path.join(
                model_base, 
                selected_pair,
                f"layer{selected_layer}", 
                f"{component}_gemini_labels.json"
            )
            
            if not os.path.exists(labels_file):
                st.error(f"No data found for {labels_file}")
                return
            
            with open(labels_file, 'r') as f:
                labels = json.load(f)
            
            # Load cluster sentences
            cluster_sentences = load_cluster_sentences(
                os.path.join(model_base, selected_pair),
                selected_layer,
                component
            )
            
            # Display clusters
            st.header(f"{component.title()} Clusters - Layer {selected_layer}")
            
            cluster_ids = [cluster_id for item in labels for cluster_id in item.keys()]
            
            # Use selectbox for cluster selection and store the actual selected value
            selected_cluster = st.selectbox(
                "Select Cluster", 
                cluster_ids, 
                index=st.session_state.current_cluster_index,
                key="cluster_selector"  # Add a unique key
            )
            
            # Display selected cluster
            for item in labels:
                if selected_cluster in item:
                    cluster_data = item[selected_cluster]
                    context_sentences = cluster_sentences.get(selected_cluster, [])
                    display_cluster_info(
                        cluster_data, 
                        f"{model_name}/{selected_pair}",
                        selected_layer,
                        selected_cluster,
                        sentences=context_sentences
                    )
                    break  # Add break to stop after finding the correct cluster
    elif view == "Aligned Clusters" and model_name not in ["coderosetta_mlm_mixed", "coderosetta_aer_mixed"]:
        display_aligned_clusters(model_base, selected_pair, selected_layer)

if __name__ == "__main__":
    main()