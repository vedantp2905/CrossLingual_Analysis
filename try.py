def format_cuda_code(input_text):
    # Split by '<CUDA>' and filter out empty strings
    cuda_statements = [s.strip() for s in input_text.split('<CUDA>') if s.strip()]
    
    # Join the statements with newlines (without adding back the <CUDA> prefix)
    formatted_text = '\n'.join(cuda_statements)
    
    return formatted_text

# Example usage
with open('output.txt', 'r', encoding='utf-8') as file:
    content = file.read()

formatted_content = format_cuda_code(content)

# Write the formatted content back to file
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(formatted_content)

def compare_files(file1_path, file2_path, output_path):
    # Read both files
    with open(file1_path, 'r', encoding='utf-8') as file1:
        lines1 = file1.readlines()
    with open(file2_path, 'r', encoding='utf-8') as file2:
        lines2 = file2.readlines()
    
    # Compare lines and store differences
    differences = []
    for i in range(max(len(lines1), len(lines2))):
        # Get lines (or empty string if file is shorter)
        line1 = lines1[i].strip() if i < len(lines1) else ''
        line2 = lines2[i].strip() if i < len(lines2) else ''
        
        if line1 != line2:
            differences.append(f"Line {i+1}:\n- File1: {line1}\n+ File2: {line2}\n")
    
    # Write differences to output file
    with open(output_path, 'w', encoding='utf-8') as out_file:
        if differences:
            out_file.writelines(differences)
        else:
            out_file.write("Files are identical\n")

# Example usage
compare_files('output.txt', 'label.out', 'differences.txt')