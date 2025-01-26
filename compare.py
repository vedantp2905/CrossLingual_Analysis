import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_metrics(data):
    total_alignments = 0
    total_calign = 0
    total_colap = 0
    
    for cluster_id in data:
        # Count total alignments
        total_alignments += len(data[cluster_id]["aligned_clusters"])
        
        # Add scores
        metrics = data[cluster_id]["metrics"]
        total_calign += metrics["calign_score"]
        total_colap += metrics["colap_score"]
    
    num_clusters = len(data)
    avg_calign = total_calign / num_clusters if num_clusters > 0 else 0
    avg_colap = total_colap / num_clusters if num_clusters > 0 else 0
    
    return {
        "total_alignments": total_alignments,
        "avg_calign": avg_calign,
        "avg_colap": avg_colap,
        "num_clusters": num_clusters
    }

def compare_all_layers():
    base_path = 'CrossLingual_Analysis/t5/cpp_cuda'
    results = {}
    
    # Process layers 0 to 13
    for layer in range(14):
        layer_path = f"{base_path}/layer{layer}"
        awesome_file = f"{layer_path}/cluster_alignments.json"
        fast_file = f"{layer_path}/cluster_alignments_fast_align.json"
        

        if os.path.exists(awesome_file) and os.path.exists(fast_file):
            # Load and calculate metrics for both files
            awesome_data = load_json_file(awesome_file)
            fast_data = load_json_file(fast_file)
            


            awesome_metrics = calculate_metrics(awesome_data)
            fast_metrics = calculate_metrics(fast_data)
            

            # Store results for this layer
            results[f"layer{layer}"] = {
                "awesome_alignments": awesome_metrics,
                "fast_alignments": fast_metrics
            }
    

    # Write results to JSON file
    output_file = 'CrossLingual_Analysis/alignment_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_comparison_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def create_comparison_plots():
    # Load the data
    data = load_comparison_data('CrossLingual_Analysis/alignment_comparison.json')
    
    # Extract layer numbers and metrics
    layers = sorted([int(layer.replace('layer', '')) for layer in data.keys()])
    
    # Prepare data for plotting
    regular_alignments = []
    fast_alignments = []
    regular_calign = []
    fast_calign = []
    regular_colap = []
    fast_colap = []
    regular_clusters = []
    fast_clusters = []
    
    for layer in layers:
        layer_data = data[f'layer{layer}']
        
        # Regular alignments data
        regular = layer_data['awesome_alignments']
        regular_alignments.append(regular['total_alignments'])
        regular_calign.append(regular['avg_calign'])
        regular_colap.append(regular['avg_colap'])
        regular_clusters.append(regular['num_clusters'])
        

        # Fast alignments data
        fast = layer_data['fast_alignments']
        fast_alignments.append(fast['total_alignments'])
        fast_calign.append(fast['avg_calign'])
        fast_colap.append(fast['avg_colap'])
        fast_clusters.append(fast['num_clusters'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot total alignments
    ax1.plot(layers, regular_alignments, 'b-', label='Awesome Alignments')
    ax1.plot(layers, fast_alignments, 'r--', label='Fast Alignments')
    ax1.set_title('Total Alignments by Layer')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Alignments')
    ax1.set_xticks(layers)
    ax1.legend()
    ax1.grid(True)
    
    # Plot average CALIGN scores
    ax2.plot(layers, regular_calign, 'b-', label='Awesome Alignments')
    ax2.plot(layers, fast_calign, 'r--', label='Fast Alignments')
    ax2.set_title('Average CALIGN Score by Layer')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Average CALIGN Score')
    ax2.set_xticks(layers)
    ax2.legend()
    ax2.grid(True)
    
    # Plot average COLAP scores
    ax3.plot(layers, regular_colap, 'b-', label='Awesome Alignments')
    ax3.plot(layers, fast_colap, 'r--', label='Fast Alignments')
    ax3.set_title('Average COLAP Score by Layer')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Average COLAP Score')
    ax3.set_xticks(layers)
    ax3.legend()
    ax3.grid(True)
    
    # Plot number of clusters
    ax4.plot(layers, regular_clusters, 'b-', label='Awesome Alignments')
    ax4.plot(layers, fast_clusters, 'r--', label='Fast Alignments')
    ax4.set_title('Number of Encoder Clusters by Layer')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Number of Clusters')
    ax4.set_xticks(layers)
    ax4.legend()
    ax4.grid(True)
    

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('CrossLingual_Analysis/alignment_comparison_plots.png')
    plt.close()

# Run the comparison
compare_all_layers()

if __name__ == "__main__":
    create_comparison_plots()