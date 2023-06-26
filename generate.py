import os
import sys
import builtins
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Set the environment variable
os.environ["HF_REMOTES_OFFLINE"] = "1"

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "checkpoints/tiiuae/falcon-40b-instruct"  # Specify the path to the downloaded model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Patch the built-in input function to return 'y' automatically
def mock_input(prompt=None):
    return 'y'

# Patch the input function to use the mock_input function
builtins.input = mock_input

try:
    # Attempt to load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__


prompt = "Write a grade 1 Addition question and corresponding equation to solve the problem."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

