import torch

def perplexity(sentence:str, tokenizer, model, stride:int=512) -> float:
    encodings = tokenizer(sentence, return_tensors='pt').input_ids
    max_length = model.config.n_positions  # 1024
    lls = []
    for i in range(0, encodings.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings[:,begin_loc:end_loc]
        target_ids = input_ids.clone()

        target_ids[:,:-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len
        
        lls.append(log_likelihood)
    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    # print(ppl)
    return ppl.item()