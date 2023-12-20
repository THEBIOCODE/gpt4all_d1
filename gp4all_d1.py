#pip install -q bitsandbytes datasets loralib sentencepiece git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/peft.git gradio==3.20.0 tenacity
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import textwrap

peft_model_id = "nomic-ai/gpt4all-lora"
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, 
    return_dict=True, 
    load_in_8bit=True, 
    device_map='auto')

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id)

def gpt4all_generate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].cuda()

    generation_config = GenerationConfig(
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    print("Generating...")
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        # return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )

    wrapped_text = textwrap.fill(tokenizer.decode(generation_output[0]), width=100)
    
    return wrapped_text

prompt = "Tell me about yourself."

history = f"""Below is a history of instructions that describe tasks, paired with an input that provides further context. Write a response that appropriately completes the request by remembering the conversation history.

### Instruction: {prompt}

### Response:"""

gen_text = gpt4all_generate(history)

print(gen_text.replace(history, ''))

prompt = "한글로 번역해주세요."

history = gen_text + f"""

### Instruction: {prompt}

### Response:"""

gen_text = gpt4all_generate(history)

print(gen_text.replace(history, ''))
