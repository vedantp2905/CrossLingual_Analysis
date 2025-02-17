import argparse
import time
import json
import os
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.util import ngrams
import numpy as np
import re
from collections import Counter
from collections import defaultdict
from itertools import chain
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

def load_sentences_and_labels(sentence_file, label_file):
    sentences = []
    labels = []

    with open(sentence_file, 'r') as f_sentences, open(label_file, 'r') as f_labels:
        for sentence, label_line in zip(f_sentences, f_labels):
            sentences.append(sentence.strip())
            labels.append(label_line.strip())

    return sentences, labels

import re

def load_clusters(cluster_file):
    """Load cluster data and print lines where the word is '|' or '||'."""
    clusters = []

    with open(cluster_file, 'r') as f_clusters:
        for line in f_clusters:
            # Strip whitespace and count the total number of '|' characters
            stripped_line = line.strip()
            pipe_count = stripped_line.count('|')

            # Determine how to handle the line based on the pipe count
            if pipe_count == 13:
                # Word is '|'
                word = '|'
            elif pipe_count == 14:
                # Word is '||'
                word = '||'
            elif pipe_count == 12:
                # Normal case: Split by '|||'
                parts = stripped_line.split('|||')
                word = parts[0]  # Normal word extraction
            else:
                print(f"Unexpected pipe count ({pipe_count}): {stripped_line}")
                continue  # Skip malformed lines

            # Extract other values from the split
            parts = stripped_line.split('|||')
            word_frequency = parts[1]
            sentence_index = int(parts[2])
            word_index = int(parts[3])
            cluster_id = parts[4].split()[-1]

            # Add the extracted data to the clusters list
            clusters.append((word, word_frequency, sentence_index, word_index, cluster_id))

    print(f"Total Clusters Loaded: {len(clusters)}")
    return clusters






def filter_label_map(label_map):
    filtered_label_map = {}
    unique_labels = set()
    
    for tag, word_list in label_map.items():
        if len(word_list) >= 6:
            filtered_label_map[tag] = set(word_list)
            unique_labels.add(tag)
            
    return filtered_label_map, unique_labels

def create_label_map_2(sentences, labels):
    label_map = {}
    unique_labels = set()

    for sentence_index, label_line in enumerate(labels):
        label_tokens = label_line.split()
        word_tokens = sentences[sentence_index].split()

        for word_index, label in enumerate(label_tokens):
            cluster_words = []

            if label in label_map:
                cluster_words = label_map[label]

            cluster_words.append(word_tokens[word_index])
            label_map[label] = cluster_words
            unique_labels.add(label)

    return filter_label_map(label_map)

def extract_words_items(cluster_words):
    word_items = [item[0] for item in cluster_words]
    return word_items

def assign_labels_to_clusters_2(label_map, clusters, threshold, sentences):
    ambiguous_clusters = {}
    non_ambiguous_clusters = {}
    none_clusters = {}
    cluster_tokens = {}  # Dictionary to hold tokens for all clusters
    g_c = group_clusters(clusters)
    ambiguous_labels = {"IDENT", "KEYWORD", "STRING", "MODIFIER", "TYPE", "NUMBER"}

    for cluster_id, cluster_words in g_c:
        word_items = extract_words_items(cluster_words)
        best_match = 0
        best_label = None

        for label_id, label_words in label_map.items():
            x = [value for value in word_items if value in label_words]
            match = len(x) / len(word_items) if word_items else 0

            if match > best_match:
                best_match = match
                best_label = label_id

        # Store tokens for all clusters
        cluster_tokens[cluster_id] = word_items

        if best_match >= threshold:
            if best_label in ambiguous_labels:
                ambiguous_clusters[cluster_id] = best_label
            else:
                non_ambiguous_clusters[cluster_id] = best_label
        else:
            none_clusters[cluster_id] = "NONE"

    print(f"Total clusters: {len(g_c)}")
    print(f"Ambiguous clusters: {len(ambiguous_clusters)}")
    print(f"Non-ambiguous clusters: {len(non_ambiguous_clusters)}")
    print(f"None clusters: {len(none_clusters)}")
    return ambiguous_clusters, non_ambiguous_clusters, none_clusters, cluster_tokens

def create_label_map(sentences, labels):
    label_map = {}
    unique_labels = set()

    for sentence_index, label_line in enumerate(labels):
        label_tokens = label_line.split()
        word_tokens = sentences[sentence_index].split()

        # Ensure the number of tokens matches
        if len(label_tokens) != len(word_tokens):
            raise ValueError(f"Token count mismatch at line {sentence_index}")

        for word_index, (word, label) in enumerate(zip(word_tokens, label_tokens)):
            key = (word_index, word)
            if key not in label_map:
                label_map[key] = []
            label_map[key].append(label)
            unique_labels.add(label)

    return label_map, unique_labels


def group_clusters(clusters):
    cluster_groups = {}

    for cluster in clusters:
        _, _, _, _, cluster_id = cluster
        if cluster_id in cluster_groups:
            cluster_groups[cluster_id].append(cluster)
        else:
            cluster_groups[cluster_id] = [cluster]

    return cluster_groups.items()

def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def longest_common_substring(s1, s2):
    # Removed this entire method
    pass

def n_gram_overlap(s1, s2, n=3):
    # Convert input to strings if they're not already
    s1 = ' '.join(s1) if isinstance(s1, list) else s1
    s2 = ' '.join(s2) if isinstance(s2, list) else s2
    
    # Generate n-grams
    s1_ngrams = set(ngrams(s1, n))
    s2_ngrams = set(ngrams(s2, n))
    
    # Calculate intersection and union
    intersection = s1_ngrams.intersection(s2_ngrams)
    union = s1_ngrams.union(s2_ngrams)
    
    # Calculate overlap
    if not union:
        return 0.0  # Return 0 if there are no n-grams
    return len(intersection) / len(union)

def character_jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    return similarity



def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj



def find_longest_common_substring(sentences):
    if not sentences:
        return ""

    # Start with the first sentence as the initial common substring
    common_substring = sentences[0]

    for sentence in sentences[1:]:
        temp_substring = ""
        for i in range(len(common_substring)):
            for j in range(i + 1, len(common_substring) + 1):
                if common_substring[i:j] in sentence and len(common_substring[i:j]) > len(temp_substring):
                    temp_substring = common_substring[i:j]
        common_substring = temp_substring

        if not common_substring:
            break

    return common_substring

def extract_patterns_from_substrings(substrings):
    pattern_counter = Counter()
    
    for substring, count in substrings.items():
        # Check for patterns using regex
        if re.match(r'^[a-z]+[A-Z]', substring):  # Lowercase followed by uppercase
            pattern_counter['lowercase-uppercase'] += count
        if re.match(r'^[a-z]+\d', substring):  # Lowercase followed by digits
            pattern_counter['lowercase-number'] += count
        if re.match(r'^[A-Z][a-z]+', substring):  # Uppercase followed by lowercase
            pattern_counter['uppercase-lowercase'] += count
        if re.match(r'^[a-z]+$', substring):  # All lowercase
            pattern_counter['all-lowercase'] += count
        if re.match(r'^[A-Z]+$', substring):  # All uppercase
            pattern_counter['all-uppercase'] += count
        # Add more regex patterns as needed

    return pattern_counter

def discover_dynamic_patterns(substrings):
    pattern_counter = Counter()
    
    for substring, count in substrings.items():
        # Analyze the structure of the substring
        structure = []
        for char in substring:
            if char.islower():
                structure.append('l')  # lowercase
            elif char.isupper():
                structure.append('u')  # uppercase
            elif char.isdigit():
                structure.append('d')  # digit
            else:
                structure.append('s')  # special character (if any)

        # Join the structure to form a pattern representation
        pattern_representation = ''.join(structure)
        pattern_counter[pattern_representation] += count

    return pattern_counter

def analyze_prefixes_suffixes(words, n=3):
    prefixes = Counter()
    suffixes = Counter()
    for word in words:
        if len(word) >= n:
            prefixes[word[:n]] += 1
            suffixes[word[-n:]] += 1
    return dict(prefixes), dict(suffixes)

def detect_abbreviations(tokens):
    """Detect likely abbreviations in tokens."""
    common_abbrevs = {
        'str': 'string',
        'num': 'number',
        'idx': 'index',
        'len': 'length',
        'tmp': 'temporary',
        'var': 'variable',
        'func': 'function',
        'param': 'parameter',
        'arr': 'array',
        'obj': 'object'
    }
    
    abbrevs = Counter()
    for token in tokens:
        # All uppercase tokens of length >= 2
        if token.isupper() and len(token) >= 2:
            abbrevs[token] += 1
        # Known abbreviations
        elif token.lower() in common_abbrevs:
            abbrevs[token] += 1
    
    return abbrevs

def analyze_word_lengths(substrings):
    word_lengths = Counter()
    for word, count in substrings.items():
        word_lengths[len(word)] += count
    return word_lengths

def analyze_character_types(substrings):
    char_types = Counter()
    for word, count in substrings.items():
        for char in word:
            if char.islower():
                char_types['lowercase'] += count
            elif char.isupper():
                char_types['uppercase'] += count
            elif char.isdigit():
                char_types['digit'] += count
            else:
                char_types['special'] += count
    return char_types

def analyze_ngrams(substrings, n=2):
    ngrams = Counter()
    for word, count in substrings.items():
        word_ngrams = [''.join(gram) for gram in zip(*[word[i:] for i in range(n)])]
        ngrams.update({gram: count for gram in word_ngrams})
    return ngrams

def vowel_consonant_ratio(substrings):
    vowels = "aeiouAEIOU"
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    vowel_count = 0
    consonant_count = 0
    for word, count in substrings.items():
        for char in word:
            if char in vowels:
                vowel_count += count
            elif char in consonants:
                consonant_count += count
    return {
        "vowel_count": vowel_count,
        "consonant_count": consonant_count,
        "ratio": vowel_count / consonant_count if consonant_count > 0 else 0
    }

def parse_clusters_file(clusters_file_path):
    clusters = {}
    with open(clusters_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('|||')
            token = parts[0]
            sentence_number = int(parts[2])
            token_id = int(parts[3])
            if (sentence_number, token_id) not in clusters:
                clusters[(sentence_number, token_id)] = []
            clusters[(sentence_number, token_id)].append(token)
    return clusters

def parse_labels_file(labels_file_path):
    labels = []
    with open(labels_file_path, 'r') as file:
        for line in file:
            labels.append(line.strip().split())
    return labels

def map_labels_to_tokens(java_in_file_path, clusters, labels):
    with open(java_in_file_path, 'r') as file:
        lines = file.readlines()

    mapped_labels = []
    for line_number, line in enumerate(lines):
        tokens = line.strip().split()
        line_labels = []
        for token_id, token in enumerate(tokens):
            if (line_number, token_id) in clusters:
                cluster_tokens = clusters[(line_number, token_id)]
                token_labels = []
                for cluster_token in cluster_tokens:
                    for label_line in labels:
                        if cluster_token in label_line:
                            token_labels.append(label_line[label_line.index(cluster_token)])
                line_labels.append(token_labels)
            else:
                line_labels.append(['UNKNOWN'])  # Or some default label
        mapped_labels.append(line_labels)

    return mapped_labels

def save_mapped_labels(mapped_labels, output_file_path):
    with open(output_file_path, 'w') as file:
        for line_labels in mapped_labels:
            file.write(' '.join([' '.join(labels) for labels in line_labels]) + '\n')

def parse_clusters_file(clusters_file_path):
    clusters = {}
    with open(clusters_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('|||')
            token = parts[0]
            line_number = int(parts[2])
            column_number = int(parts[3])
            cluster_id = int(parts[4])
            clusters[(line_number, column_number)] = (token, cluster_id)
    return clusters

def parse_java_files(java_in_path, java_label_path):
    with open(java_in_path, 'r') as in_file, open(java_label_path, 'r') as label_file:
        java_in_lines = in_file.readlines()
        java_label_lines = label_file.readlines()
    return java_in_lines, java_label_lines

def map_tokens_to_labels(java_in_lines, java_label_lines, clusters):
    mapped_data = {}
    for line_number, (in_line, label_line) in enumerate(zip(java_in_lines, java_label_lines), start=1):
        tokens = in_line.strip().split()
        labels = label_line.strip().split()
        for column_number, (token, label) in enumerate(zip(tokens, labels)):
            cluster_info = clusters.get((line_number, column_number), ('UNKNOWN', 'NO_CLUSTER'))
            cluster_id = cluster_info[1]
            if cluster_id not in mapped_data:
                mapped_data[cluster_id] = []
            mapped_data[cluster_id].append({
                'token': token,
                'label': label
            })
    return mapped_data

def save_mapped_labels_to_json(mapped_data, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(mapped_data, file, indent=2)

def write_none_cluster_labels(none_clusters, mapped_labels_file, output_dir, method, threshold):
    output_file = f'{output_dir}/none_cluster_labels_{method}_{threshold}.json'
    
    # Load the mapped labels
    with open(mapped_labels_file, 'r') as f:
        mapped_data = json.load(f)
    
    none_cluster_data = {}
    
    for cluster_id in none_clusters:
        cluster_id_str = str(cluster_id)  # Convert to string if it's not already
        if cluster_id_str in mapped_data:
            # Count the occurrences of each label in the cluster
            label_counts = Counter(item['label'] for item in mapped_data[cluster_id_str])
            none_cluster_data[cluster_id_str] = dict(label_counts)
        else:
            none_cluster_data[cluster_id_str] = {}  # Empty dict if no data found

    # Write the NONE cluster data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(none_cluster_data, f, indent=2)

    print(f"NONE cluster labels written to {output_file}")

def count_total_labels(label_file):
    total_labels = 0
    with open(label_file, 'r') as f:
        for line in f:
            labels = line.strip().split()
            total_labels += len(labels)
    print(f"Total labels processed: {total_labels}")
    return total_labels

def count_total_tokens(sentence_file):
    total_tokens = 0
    with open(sentence_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            total_tokens += len(tokens)
    print(f"Total tokens processed: {total_tokens}")
    return total_tokens

def calculate_precision_recall(true_tokens, predicted_tokens):
    true_set = set(true_tokens)
    pred_set = set(predicted_tokens)
    
    true_positives = len(true_set.intersection(pred_set))
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def analyze_lexical_patterns_for_clusters(clusters, cluster_tokens, output_dir, method, threshold, cluster_type, lcs_occurrence_threshold=2):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    all_patterns = {}
    lcs_metrics = {}
    substring_metrics = {}

    for cluster_id in clusters:
        sentences = cluster_tokens.get(cluster_id, [])
        
        # First find the longest common substring
        longest_common_substring = find_longest_common_substring(sentences)
        
        # Convert sentences to a list of words and filter out LCS parts
        # We'll still use lcs_parts for filtering but won't save it
        if isinstance(sentences, str):
            words = sentences.split()
        else:
            words = [word for sentence in sentences for word in (sentence.split() if isinstance(sentence, str) else [sentence])]
        
        # Filter out LCS parts without storing them in the output
        filtered_words = [word for word in words 
                         if word not in (longest_common_substring.split() if longest_common_substring else [])]
        
        # Find common substrings appearing at least twice, excluding LCS parts
        common_substrings = Counter(filtered_words)
        
        # Filter common substrings based on occurrence threshold
        filtered_common_substrings = {word: count for word, count in common_substrings.items() 
                                    if count >= lcs_occurrence_threshold}
        
        # Create list of substring words that meet the threshold
        substring_words = list(filtered_common_substrings.keys())

        # Find the longest common substring
        longest_common_substring = find_longest_common_substring(sentences)

        # Extract dynamic patterns
        lcs_patterns = discover_dynamic_patterns({longest_common_substring: 1})
        common_substring_patterns = discover_dynamic_patterns(filtered_common_substrings)

        # Analyze prefixes and suffixes
        prefixes, suffixes = analyze_prefixes_suffixes(filtered_common_substrings)

        # Detect abbreviations
        abbreviations = detect_abbreviations(filtered_common_substrings)

        # Analyze word lengths
        word_lengths = analyze_word_lengths(filtered_common_substrings)

        # Analyze character types
        char_types = analyze_character_types(filtered_common_substrings)

        # Analyze n-grams
        bigrams = analyze_ngrams(filtered_common_substrings, n=2)

        # Calculate vowel-consonant ratio
        vc_ratio = vowel_consonant_ratio(filtered_common_substrings)

        # Store all metrics for this cluster
        all_patterns[cluster_id] = {
            "longest_common_substring": longest_common_substring,
            "common_substrings": filtered_common_substrings,
            "substring_words": substring_words,
            "lcs_patterns": dict(lcs_patterns),
            "common_substring_patterns": dict(common_substring_patterns),
            "common_prefixes": prefixes.most_common(5),
            "common_suffixes": suffixes.most_common(5),
            "abbreviations": abbreviations.most_common(),
            "word_length_distribution": word_lengths.most_common(),
            "character_types": dict(char_types),
            "common_bigrams": bigrams.most_common(10),
            "vowel_consonant_ratio": vc_ratio
        }

        # Calculate metrics for this cluster
        if longest_common_substring:
            lcs_precision, lcs_recall, lcs_f1 = calculate_precision_recall(words, [longest_common_substring])
            lcs_metrics[cluster_id] = {
                "precision": lcs_precision,
                "recall": lcs_recall,
                "f1": lcs_f1
            }

        substring_precision, substring_recall, substring_f1 = calculate_precision_recall(words, substring_words)
        substring_metrics[cluster_id] = {
            "precision": substring_precision,
            "recall": substring_recall,
            "f1": substring_f1
        }

    # Save all patterns to a single JSON file for the specific cluster type
    json_file_path = f'{output_dir}/lexical_patterns_{cluster_type}_clusters_{method}_{threshold}.json'
    with open(json_file_path, 'w') as f:
        json.dump(all_patterns, f, indent=2, default=numpy_to_python)

    return all_patterns, lcs_metrics, substring_metrics

def nested_defaultdict():
    return defaultdict(lambda: defaultdict(int))

def analyze_clusters_with_properties(clusters, cluster_tokens, unique_substrings_threshold):
    labeled_clusters = {}
    pattern_stats = Counter()

    for cluster_id, tokens in cluster_tokens.items():
        # Find all unique substrings in the cluster
        unique_substrings = set()
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                common_sub = find_longest_common_substring([tokens[i], tokens[j]])
                if len(common_sub) > 2:  # Only consider substrings longer than 2 characters
                    unique_substrings.add(common_sub)
        
        # Check if the number of unique substrings meets the threshold
        if len(unique_substrings) >= unique_substrings_threshold:
            labeled_clusters[cluster_id] = 'SUBSTRING_PATTERN'
            pattern_stats['SUBSTRING_PATTERN'] += 1
        else:
            labeled_clusters[cluster_id] = 'NO_PATTERN'
            pattern_stats['NO_PATTERN'] += 1

    return labeled_clusters, pattern_stats

def save_filtered_clusters(labeled_clusters, cluster_tokens, output_dir, unique_substrings_threshold, cluster_type):
    # Filter clusters based on SUBSTRING_PATTERN label
    filtered_clusters = {}
    for cluster_id, tokens in cluster_tokens.items():
        if labeled_clusters.get(cluster_id) == 'SUBSTRING_PATTERN':
            # Find unique substrings for this cluster
            unique_substrings = set()
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    common_sub = find_longest_common_substring([tokens[i], tokens[j]])
                    if len(common_sub) > 2:
                        unique_substrings.add(common_sub)
            
            if len(unique_substrings) >= unique_substrings_threshold:
                filtered_clusters[cluster_id] = {
                    "unique_substrings": list(unique_substrings),
                    "substring_count": len(unique_substrings)
                }
    
    # Count the number of clusters that meet the threshold
    total_meeting_threshold = len(filtered_clusters)

    # Debugging: Print the number of clusters being saved
    print(f"Saving {total_meeting_threshold} {cluster_type} clusters to JSON.")

    # Prepare the data to be saved
    data_to_save = {
        "total_clusters_meeting_threshold": total_meeting_threshold,
        "clusters": filtered_clusters
    }
    
    output_file = os.path.join(output_dir, f'filtered_{cluster_type}_clusters_substrings_{unique_substrings_threshold}.json')
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Filtered {cluster_type} clusters saved to {output_file}")

def generate_consolidated_summary(ambiguous_patterns, non_ambiguous_patterns, none_patterns, 
                                  ambiguous_lcs_metrics, non_ambiguous_lcs_metrics, none_lcs_metrics,
                                  ambiguous_substring_metrics, non_ambiguous_substring_metrics, none_substring_metrics,
                                  output_dir, method, threshold):
    summary = {
        'Metric': [],
        'Ambiguous': [],
        'Non-Ambiguous': [],
        'None': []
    }

    def add_metric(name, amb_values, non_amb_values, none_values):
        for stat in ['Mean', 'Median', 'Std Dev']:
            summary['Metric'].append(f'{name} ({stat})')
            if stat == 'Mean':
                summary['Ambiguous'].append(np.mean(amb_values) if amb_values else float('nan'))
                summary['Non-Ambiguous'].append(np.mean(non_amb_values) if non_amb_values else float('nan'))
                summary['None'].append(np.mean(none_values) if none_values else float('nan'))
            elif stat == 'Median':
                summary['Ambiguous'].append(np.median(amb_values) if amb_values else float('nan'))
                summary['Non-Ambiguous'].append(np.median(non_amb_values) if non_amb_values else float('nan'))
                summary['None'].append(np.median(none_values) if none_values else float('nan'))
            else:  # Std Dev
                summary['Ambiguous'].append(np.std(amb_values) if amb_values else float('nan'))
                summary['Non-Ambiguous'].append(np.std(non_amb_values) if non_amb_values else float('nan'))
                summary['None'].append(np.std(none_values) if none_values else float('nan'))

    # Add LCS length metrics
    amb_lcs_lengths = [len(cluster.get('longest_common_substring', '')) for cluster in ambiguous_patterns.values()]
    non_amb_lcs_lengths = [len(cluster.get('longest_common_substring', '')) for cluster in non_ambiguous_patterns.values()]
    none_lcs_lengths = [len(cluster.get('longest_common_substring', '')) for cluster in none_patterns.values()]
    add_metric('LCS Length', amb_lcs_lengths, non_amb_lcs_lengths, none_lcs_lengths)

    # Add all metrics (same as before)
    # Substring metrics
    for metric in ['precision', 'recall', 'f1']:
        amb_values = [m[metric] for m in ambiguous_substring_metrics.values()]
        non_amb_values = [m[metric] for m in non_ambiguous_substring_metrics.values()]
        none_values = [m[metric] for m in none_substring_metrics.values()]
        add_metric(f'Substring {metric.capitalize()}', amb_values, non_amb_values, none_values)

    # Common substring count
    amb_counts = [len(cluster.get('common_substrings', {})) for cluster in ambiguous_patterns.values()]
    non_amb_counts = [len(cluster.get('common_substrings', {})) for cluster in non_ambiguous_patterns.values()]
    none_counts = [len(cluster.get('common_substrings', {})) for cluster in none_patterns.values()]
    add_metric('Common Substring Count', amb_counts, non_amb_counts, none_counts)

    # Substring length distribution
    amb_lengths = [len(substring) for cluster in ambiguous_patterns.values() for substring in cluster.get('common_substrings', [])]
    non_amb_lengths = [len(substring) for cluster in non_ambiguous_patterns.values() for substring in cluster.get('common_substrings', [])]
    none_lengths = [len(substring) for cluster in none_patterns.values() for substring in cluster.get('common_substrings', [])]
    add_metric('Substring Length', amb_lengths, non_amb_lengths, none_lengths)

    # LCS metrics
    for metric in ['precision', 'recall', 'f1']:
        amb_values = [m[metric] for m in ambiguous_lcs_metrics.values()]
        non_amb_values = [m[metric] for m in non_ambiguous_lcs_metrics.values()]
        none_values = [m[metric] for m in none_lcs_metrics.values()]
        add_metric(f'LCS {metric.capitalize()}', amb_values, non_amb_values, none_values)

    # Word length
    amb_lengths = [len(word) for cluster in ambiguous_patterns.values() for word in cluster.get('common_substrings', [])]
    non_amb_lengths = [len(word) for cluster in non_ambiguous_patterns.values() for word in cluster.get('common_substrings', [])]
    none_lengths = [len(word) for cluster in none_patterns.values() for word in cluster.get('common_substrings', [])]
    add_metric('Word Length', amb_lengths, non_amb_lengths, none_lengths)

    # Character type counts
    for char_type in ['lowercase', 'uppercase', 'digit', 'special']:
        amb_counts = [cluster.get('character_types', {}).get(char_type, 0) for cluster in ambiguous_patterns.values()]
        non_amb_counts = [cluster.get('character_types', {}).get(char_type, 0) for cluster in non_ambiguous_patterns.values()]
        none_counts = [cluster.get('character_types', {}).get(char_type, 0) for cluster in none_patterns.values()]
        add_metric(f'{char_type.capitalize()} Character Count', amb_counts, non_amb_counts, none_counts)

    # Common prefixes and suffixes count
    amb_prefix_counts = [len(cluster.get('common_prefixes', [])) for cluster in ambiguous_patterns.values()]
    non_amb_prefix_counts = [len(cluster.get('common_prefixes', [])) for cluster in non_ambiguous_patterns.values()]
    none_prefix_counts = [len(cluster.get('common_prefixes', [])) for cluster in none_patterns.values()]
    add_metric('Common Prefixes Count', amb_prefix_counts, non_amb_prefix_counts, none_prefix_counts)

    amb_suffix_counts = [len(cluster.get('common_suffixes', [])) for cluster in ambiguous_patterns.values()]
    non_amb_suffix_counts = [len(cluster.get('common_suffixes', [])) for cluster in non_ambiguous_patterns.values()]
    none_suffix_counts = [len(cluster.get('common_suffixes', [])) for cluster in none_patterns.values()]
    add_metric('Common Suffixes Count', amb_suffix_counts, non_amb_suffix_counts, none_suffix_counts)

    # Vowel-consonant ratio
    amb_ratios = [cluster.get('vowel_consonant_ratio', {}).get('ratio', float('nan')) for cluster in ambiguous_patterns.values()]
    non_amb_ratios = [cluster.get('vowel_consonant_ratio', {}).get('ratio', float('nan')) for cluster in non_ambiguous_patterns.values()]
    none_ratios = [cluster.get('vowel_consonant_ratio', {}).get('ratio', float('nan')) for cluster in none_patterns.values()]
    add_metric('Vowel-Consonant Ratio', amb_ratios, non_amb_ratios, none_ratios)

    # Create a DataFrame
    df = pd.DataFrame(summary)

    # Generate formatted table
    formatted_table = tabulate(df, headers='keys', tablefmt='pipe', floatfmt='.4f')

    # Save CSV file
    csv_file_path = os.path.join(output_dir, f'lexical_analysis_summary.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Consolidated summary CSV saved to {csv_file_path}")

    return formatted_table, csv_file_path

def extract_impl_from_path(path):
    """Extract implementation name from the path by taking the last directory name."""
    # Remove trailing slash if present
    path = path.rstrip('/')
    # Get the last directory name
    impl = os.path.basename(path)
    if not impl:
        raise ValueError("Could not extract implementation from path. Invalid path format.")
    return impl

def analyze_lexical_patterns(cluster_file, output_dir):
    """Analyze patterns in cluster file without saving individual results"""
    clusters = defaultdict(list)
    cluster_frequencies = defaultdict(int)
    
    with open(cluster_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped_line = line.strip()
            pipe_count = stripped_line.count('|')

            # First handle the word and frequency based on pipe count
            if pipe_count == 13:
                word = '|'
                parts = stripped_line.split('|||')
                frequency = int(parts[1].replace('|', ''))
            elif pipe_count == 14:
                word = '||'
                parts = stripped_line.split('|||')
                frequency = int(parts[1].replace('||', ''))
            elif pipe_count == 12:
                parts = stripped_line.split('|||')
                word = parts[0]
                frequency = int(parts[1])
            else:
                print(f"Unexpected pipe count ({pipe_count}): {stripped_line}")
                continue

            # Extract other values
            sentence_index = int(parts[2])
            word_index = int(parts[3])
            cluster_id = parts[4].split()[-1]
            
            # Add word and its frequency to appropriate cluster
            clusters[cluster_id].append(word)
            cluster_frequencies[cluster_id] += frequency
    
    # Analyze patterns for each cluster
    results = {}
    for cluster_id, words in clusters.items():
        # Find common substrings appearing at least twice
        common_substrings = Counter(words)
        filtered_common_substrings = {word: count for word, count in common_substrings.items() 
                                    if count >= 2}  # Using threshold of 2
        
        # Find longest common substring
        lcs = find_longest_common_substring(words)
        
        # Calculate word lengths
        word_lengths = Counter([len(word) for word in filtered_common_substrings.keys()])
        
        # Calculate character types
        char_types = Counter()
        for word in filtered_common_substrings.keys():
            for char in word:
                if char.islower():
                    char_types['lowercase'] += 1
                elif char.isupper():
                    char_types['uppercase'] += 1
                elif char.isdigit():
                    char_types['digit'] += 1
                else:
                    char_types['special'] += 1

        # Calculate vowel-consonant ratio
        vc_ratio = vowel_consonant_ratio(filtered_common_substrings)

        results[cluster_id] = {
            'tokens': list(filtered_common_substrings.keys()),  # Only common tokens
            'unique_token_count': len(filtered_common_substrings),
            'total_token_count': cluster_frequencies[cluster_id],
            'longest_common_substring': lcs,
            'word_lengths': dict(word_lengths),
            'character_types': dict(char_types),
            'vowel_consonant_ratio': vc_ratio
        }
    
    return results

def detect_patterns(words):
    patterns = Counter()
    for word in words:
        structure = []
        for char in word:
            if char.islower():
                structure.append('a')
            elif char.isupper():
                structure.append('A')
            elif char.isdigit():
                structure.append('1')
            else:
                structure.append('_')
        pattern = ''.join(structure)
        patterns[pattern] += 1
    return dict(patterns)

def find_cluster_files(base_path):
    """Find cluster files matching the pattern layer{N}/kmeans/sklearn/clusters-kmeans-500.txt"""
    cluster_files = []
    for layer_dir in os.listdir(base_path):
        if not layer_dir.startswith('layer'):  # Only process directories starting with 'layer'
            continue
            
        layer_path = os.path.join(base_path, layer_dir)
        if os.path.isdir(layer_path):
            cluster_file = os.path.join(layer_path, 'decoder-clusters-kmeans-500.txt')
            if os.path.isfile(cluster_file):
                cluster_files.append(cluster_file)
                print(f"Found cluster file: {cluster_file}")  # Debug print
    
    return cluster_files

def create_layer_comparison_graphs(all_layer_results, output_dir):
    """Create line graphs comparing metrics across layers"""
    plt.style.use('default')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
    
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['lines.linewidth'] = 2
    
    def safe_mean(values):
        values = [v for v in values if v is not None and not np.isnan(v) and v > 0]
        if not values:
            return 0
        return np.mean(values)
    
    # Print all keys to debug
    print(f"Available keys in all_layer_results: {list(all_layer_results.keys())}")
    
    # Filter and sort layers numerically - handle both 'layer' and 'layerN' formats
    layers = []
    for key in all_layer_results.keys():
        if 'layer' in key.lower():
            # Extract the number from the layer name
            num = ''.join(filter(str.isdigit, key))
            if num:  # Only add if we found a number
                layers.append(f"layer{num}")
    
    # Sort layers by their numeric value
    layers.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    if not layers:
        print("ERROR: No valid layer names found!")
        return
        
    x_values = [int(''.join(filter(str.isdigit, layer))) for layer in layers]
    
    print(f"Processing layers: {layers}")
    print(f"X values: {x_values}")
    
    # Define all metrics to plot
    metrics = [
        ('unique_token_count', 'Average Unique Tokens per Cluster', 'Number of Tokens'),
        ('total_token_count', 'Average Total Token Occurrences per Cluster', 'Number of Occurrences'),
        ('word_length', 'Average Word Length', 'Characters'),
        ('char_type_distribution', 'Character Type Distribution', 'Count'),
        ('lcs_length', 'Average Longest Common Substring Length', 'Characters'),
        ('vowel_consonant_ratio', 'Average Vowel-Consonant Ratio', 'Ratio')
    ]
    
    for metric, title, ylabel in metrics:
        print(f"\nProcessing metric: {metric}")
        plt.figure()
        
        if metric == 'char_type_distribution':
            char_types = ['lowercase', 'uppercase', 'digit', 'special']
            for i, char_type in enumerate(char_types):
                y_values = []
                for layer in layers:
                    values = [cluster['character_types'].get(char_type, 0) 
                             for cluster in all_layer_results[layer].values()]
                    avg = np.mean(values) if values else 0
                    y_values.append(avg)
                plt.plot(x_values, y_values, marker='o', label=char_type.capitalize(), color=colors[i % len(colors)])
            plt.legend()
        else:
            y_values = []
            for layer in layers:
                if metric == 'word_length':
                    lengths = []
                    for cluster in all_layer_results[layer].values():
                        lengths.extend([len(token) for token in cluster['tokens']])
                    avg = np.mean(lengths) if lengths else 0
                elif metric == 'lcs_length':
                    values = [len(cluster['longest_common_substring']) 
                             for cluster in all_layer_results[layer].values()]
                    avg = np.mean(values) if values else 0
                elif metric == 'vowel_consonant_ratio':
                    values = [cluster['vowel_consonant_ratio'].get('ratio', 0) 
                             for cluster in all_layer_results[layer].values()]
                    avg = np.mean(values) if values else 0
                else:  # Standard metrics (token counts)
                    values = [cluster.get(metric, 0) for cluster in all_layer_results[layer].values()]
                    avg = np.mean(values) if values else 0
                y_values.append(avg)
            plt.plot(x_values, y_values, marker='o', color=colors[0])
        
        plt.title(title)
        plt.xlabel('Layer')
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        output_path = os.path.join(output_dir, f'layer_comparison_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Perform lexical analysis on code tokens")
    parser.add_argument("--layers-dir", required=True, 
                       help="Base directory containing all layer directories")
    
    args = parser.parse_args()
    
    # Find all cluster files recursively
    cluster_files = find_cluster_files(args.layers_dir)
    
    if not cluster_files:
        print(f"No cluster files found in {args.layers_dir}")
        return
    
    print(f"Found {len(cluster_files)} cluster files")
    
    # Store results for all layers
    all_layer_results = {}
    
    # Process each cluster file
    for cluster_file in cluster_files:
        # Extract layer name directly from the file path
        layer_name = os.path.basename(os.path.dirname(cluster_file))  # This will get 'layerN'
        
        print(f"\nProcessing cluster file: {cluster_file}")
        
        try:
            # Run analysis for this cluster file
            results = analyze_lexical_patterns(cluster_file, None)
            all_layer_results[layer_name] = results  # Store results with layer name as key
            print(f"Successfully analyzed {len(results)} clusters")
        except Exception as e:
            print(f"Error processing {cluster_file}: {str(e)}")
            continue
    
    # Create and save layer comparison graphs
    graphs_dir = os.path.join(args.layers_dir, 'layer_comparison_graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Debug print to verify layer results
    print(f"\nLayer results collected for: {list(all_layer_results.keys())}")
    
    create_layer_comparison_graphs(all_layer_results, graphs_dir)
    print(f"\nLayer comparison graphs saved to: {graphs_dir}")

if __name__ == "__main__":
    main()
