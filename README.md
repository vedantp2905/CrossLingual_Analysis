# Code Concept Cluster Explorer

A Streamlit-based web application for exploring and evaluating code concept clusters in neural machine translation models. This tool allows researchers to analyze encoder-decoder clusters, evaluate their alignments, and store evaluations in a Supabase database.

## Features

- **Dual View Modes**:
  - Individual Clusters: Examine encoder or decoder clusters separately
  - Aligned Clusters: Analyze and evaluate encoder-decoder cluster pairs

- **Visualization**:
  - Word clouds for token visualization
  - Highlighted context sentences
  - Metadata display for syntactic and semantic information

- **Evaluation Capabilities**:
  - Syntactic accuracy assessment
  - Semantic accuracy assessment
  - Alignment evaluation between encoder-decoder pairs
  - Note-taking functionality

## Prerequisites

- Python 3.x
- Streamlit
- Supabase account and credentials
- Required Python packages:
  ```bash
  pip install streamlit supabase-py python-dotenv matplotlib wordcloud plotly
  ```

## Setup

1. Clone the repository
2. Create a `.env` file with your Supabase credentials:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

3. Ensure your data directory structure follows the pattern:
   ```
   model_directory/
   ├── language_pair/
   │   ├── layer0/
   │   │   ├── encoder-clusters-kmeans-500.txt
   │   │   ├── decoder-clusters-kmeans-500.txt
   │   │   ├── encoder_gemini_labels.json
   │   │   ├── decoder_gemini_labels.json
   │   │   └── Alignments_with_LLM_labels_layer0.json
   │   ├── input.in
   │   └── label.out
   ```

## Running the Application

```bash
streamlit run app.py
```

## Usage

1. Enter the model directory path in the sidebar (t5/coderosetta) 
2. Select a language pair
3. Choose a layer number
4. Select either encoder or decoder component
5. Choose between Individual or Aligned Clusters view
6. Evaluate clusters and submit assessments

## Database Schema

### Cluster Evaluations Table
- model (string)
- language_pair (string)
- layer_number (integer)
- cluster_id (string)
- syntactic_accuracy (string)
- semantic_accuracy (string)
- notes (text)

### Alignment Evaluations Table
- model (string)
- language_pair (string)
- layer_number (integer)
- encoder_cluster_id (string)
- decoder_cluster_id (string)
- alignment_accurate (boolean)
- alignment_types (array)
- other_notes (text)
- notes (text)