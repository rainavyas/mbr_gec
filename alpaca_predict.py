'''
Generate output predictions file with ids using the alpaca model
'''

import argparse
import sys
import os

from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import torch

def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input_ctxt}

        ### Response:"""
    
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:"""


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/alpaca_predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
    model = LlamaForCausalLM.from_pretrained(
        "chainyo/alpaca-lora-7b",
        # load_in_8bit=True,
        # torch_dtype=torch.float16,
        # device_map="auto",
    )
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
    )

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    instruction = "Grammatically correct the following?"
    input_ctxt = 'I is run in park.'
    prompt = generate_prompt(instruction, input_ctxt)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )

    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    import pdb; pdb.set_trace()
    print(response)

