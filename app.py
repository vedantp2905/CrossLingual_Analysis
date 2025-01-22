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

def display_cluster_info(cluster_data, model_pair: str, layer_number: int, cluster_id: str, sentences=None):
    """Display cluster information including word cloud, metadata and sentences"""
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
    # Determine file paths - using correct filename pattern with kmeans and cluster count
    cluster_file = os.path.join(model_dir, f"layer{layer}", f"{component}-clusters-kmeans-500.txt")
    sentence_file = os.path.join(model_dir, "input.in" if component == "encoder" else "label.out")
    
    # print(f"Loading cluster file: {cluster_file}")  # Debug print
    # print(f"Loading sentence file: {sentence_file}")  # Debug print
    
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
    
    # Display alignment metrics - fixed to use correct key structure
    st.write("### Alignment Metrics")
    if metrics_encoder_id in alignment_metrics:
        metrics = alignment_metrics[metrics_encoder_id]["metrics"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Match Percentage", f"{metrics['match_percentage']:.2%}")
            st.metric("Source Cluster Size", metrics["source_cluster_size"])
        
        with col2:
            st.metric("Aligned Word Count", metrics["aligned_word_count"])
            st.metric("Total Words", metrics["total_words"])
            
        with col3:
            st.metric("Size Threshold", metrics["size_threshold"])
            st.metric("Translation Threshold", metrics["translation_threshold"])
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
            html += f"<span style='color: #2196F3; font-weight: bold;'>{token}</span> "
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
    """Returns a list of layer numbers that have valid alignment files."""
    layers = []
    pair_dir = os.path.join(model_base, selected_pair)
    
    # Only include layers that have the alignment file
    for item in os.listdir(pair_dir):
        if item.startswith('layer'):
            layer_num = int(item.replace('layer', ''))
            alignment_file = os.path.join(
                pair_dir, 
                item, 
                f"Alignments_with_LLM_labels_layer{layer_num}.json"
            )
            if os.path.isfile(alignment_file):
                # Optionally validate JSON content
                try:
                    with open(alignment_file, 'r', encoding='utf-8') as f:
                        json.load(f)
                    layers.append(layer_num)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    
    return sorted(layers)

def validate_selected_layer(layer: int, available_layers: List[int]) -> int:
    """Validates and returns a valid layer number."""
    if not available_layers:
        raise ValueError("No valid layers found")
    
    if layer not in available_layers:
        # Return the first available layer if selected layer is invalid
        return available_layers[0]
    
    return layer

def main():
    st.set_page_config(layout="wide", page_title="Code Concept Explorer")
    
    st.title("Code Concept Cluster Explorer")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Model selection dropdown instead of text input
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["t5", "coderosetta"]
    )
    model_base = os.path.join( model_name)
    
    # Get available language pairs
    lang_pairs = [d for d in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, d))]
    if not lang_pairs:
        st.error("No language pairs found in the specified directory")
        return
        
    selected_pair = st.sidebar.selectbox("Language Pair", lang_pairs)
    
    # Add view selection
    view = st.sidebar.radio(
        "View", 
        ["Individual Clusters", "Aligned Clusters", "Top Semantic Tags"]
    )
    
    # Initialize session state for cluster index if not exists
    if 'current_cluster_index' not in st.session_state:
        st.session_state.current_cluster_index = 0
    
    if view == "Top Semantic Tags":
        display_top_semantic_tags(model_base, selected_pair)
    else:
        # Get available layers
        available_layers = get_available_layers(model_base, selected_pair)
        
        # Get and validate selected layer
        try:
            selected_layer = int(st.sidebar.selectbox(
                "Layer",
                range(len(available_layers)),
                format_func=lambda x: f"Layer {available_layers[x]}"
            ))
        except ValueError:
            selected_layer = 0
        
        selected_layer = validate_selected_layer(selected_layer, available_layers)
        
        # Component selection (only show for Individual Clusters view)
        if view == "Individual Clusters":
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
            
            # Use session state for cluster selection
            if st.session_state.current_cluster_index >= len(cluster_ids):
                st.session_state.current_cluster_index = 0
            
            selected_cluster = cluster_ids[st.session_state.current_cluster_index]
            st.selectbox("Select Cluster", cluster_ids, index=st.session_state.current_cluster_index)
            
            # Display selected cluster
            for item in labels:
                if selected_cluster in item:
                    cluster_data = item[selected_cluster]
                    context_sentences = cluster_sentences.get(selected_cluster, [])
                    display_cluster_info(
                        cluster_data, 
                        f"{model_name}/{selected_pair}",
                        selected_layer,
                        selected_cluster,  # Pass cluster ID directly
                        sentences=context_sentences
                    )
        else:
            display_aligned_clusters(model_base, selected_pair, selected_layer)

if __name__ == "__main__":
    main()