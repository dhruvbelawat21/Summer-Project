import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

tokenizer.pad_token = tokenizer.eos_token

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model is running on {device}")

model.to(device)

prompt = """
        In the context of 3GPP Release 17, when a supporting UE reattaches to the same
        core network operator in a shared network, it can request the core network node
        to remember its previous selection. This enables seamless reattachment without the need for a fresh attach procedure, streamlining the process and maintaining
        continuity of service for the user.
        Based on the above information, answer the question below.
        "question": "How does a supporting UE attach to the same core network operator from which it detached in a shared network? [3GPP Release 17]",
"option 1": "It requests the core network node to remember its previous selection.",
"option 2": "It uses information stored in the UE when it was detached.",
"option 3": "It relies on the SIM/USIM card for information.",
"option 4": "It performs a fresh attach procedure.",
Answer with the option number only: 1, 2, 3, or 4.
"""

prompt2 = """
      "Question": "How does a supporting UE attach to the same core network operator from which it detached in a shared network?",
      "option 1": "It requests the core network node to remember its previous selection.",
      "option 2": "It uses information stored in the UE when it was detached.",
      "option 3": "It relies on the SIM/USIM card for information.",
      "option 4": "It performs a fresh attach procedure.",
      Answer with the option number only: 1, 2, 3, or 4.
"""

prompt3 = "Where does the sun rise?"

# Tokenize the input prompt with padding and attention mask

tokenized_input = tokenizer(prompt2, return_tensors="pt", add_special_tokens=False, padding=True)

# Move the tokenized inputs to the appropriate device
input_ids = tokenized_input['input_ids'].to(device)
attention_mask = tokenized_input['attention_mask'].to(device)

max_new_tokens = 50

# Generate the output
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,  # Set to a very small value to encourage short output
        do_sample=True,
        temperature=0.7,  # Adjust temperature for diversity
        top_k=50,         # Use top-k sampling to limit the token pool
        top_p=0.95,       # Use top-p (nucleus) sampling
        pad_token_id=tokenizer.pad_token_id
    )

# Decode the output tokens to a string
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_answer_option(output_text):
    # Split the output text by newlines
    lines = output_text.strip().split('\n')

    # Extract the last line which contains the generated text
    last_line = lines[-1]

    # Extract the option number from the last line
    answer_option = last_line.strip()

    return answer_option

print(output)
#print(extract_answer_option(output))
