import os
import sys
import builtins
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdapterType, AdapterConfig
#from adapter-transformers import AdapterType, AdapterConfig, load_adapter

# Set the environment variable
os.environ["HF_REMOTES_OFFLINE"] = "1"

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "checkpoints/tiiuae/falcon-40b-instruct"  # Specify the path to the downloaded model
adapter_path = "output/checkpoint-3250"  # Specify the path to the adapter weights
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
        load_in_4bit=True, 
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__

# Load the adapter weights
adapter_name = "adapter_model"  # Specify the name of the adapter
adapter_config = AdapterConfig.load(adapter_path)
model.load_adapter(model, adapter_name, config=adapter_config)

prompt = "Write a grade 1 Addition question and corresponding equation to solve the problem."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1, adapter_names=[adapter_name])

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
