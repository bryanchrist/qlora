from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "checkpoints/tiiuae/falcon-40b-instruct"  # Specify the path to the downloaded model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "Write a grade 1 Addition question and corresponding equation to solve the problem."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

