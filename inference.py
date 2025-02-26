from transformers import AutoTokenizer, EncoderDecoderModel, RobertaTokenizer
import sys

# Load the CodeRosetta model and tokenizer
try:
    model = EncoderDecoderModel.from_pretrained('CodeRosetta/CodeRosetta_cpp2cuda_ft')
    tokenizer = RobertaTokenizer.from_pretrained('CodeRosetta/CodeRosetta_cpp2cuda_ft')
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    sys.exit(1)

# Read input from file
try:
    with open('coderosetta_ft/cpp_cuda/input.in', 'r', encoding='utf-8') as f:
        input_cpp_codes = f.readlines()
except Exception as e:
    print(f"Error reading input file: {e}")
    sys.exit(1)

# Process each line separately and collect results
generated_codes = []
for i, input_cpp_code in enumerate(input_cpp_codes, 1):
    print(f"Processing function {i}")
    try:
        # Skip empty lines
        if not input_cpp_code.strip():
            continue
            
        # Encode the input C++ Code
        input_ids = tokenizer.encode(input_cpp_code.strip(), return_tensors="pt")

        # Set the start token to <CUDA>
        start_token = "<CUDA>"
        decoder_start_token_id = tokenizer.convert_tokens_to_ids(start_token)

        # Generate the CUDA code
        output = model.generate(
            input_ids=input_ids, 
            decoder_start_token_id=decoder_start_token_id,
            max_length=256
        )

        # Decode the generated output
        generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_codes.append(generated_code)
        
    except Exception as e:
        print(f"Error processing line {i}: {e}")
        generated_codes.append(f"Error processing function {i}: {e}\n\n")

# Write output to file
try:
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write("// CodeRosetta C++ to CUDA Conversion Results\n\n")
        f.write("".join(generated_codes))
except Exception as e:
    print(f"Error writing output file: {e}")
    sys.exit(1)

print("Generated CUDA code has been written to output.txt")

