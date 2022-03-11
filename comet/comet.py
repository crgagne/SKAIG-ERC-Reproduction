
import torch
from torch.nn import functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import numpy as np

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Comet:
    '''Interface for generating text and extracting hidden states from pretrained COMET'''
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def get_hidden(self, queries, tok):
        tok_enc = self.tokenizer.encode(tok)[1] # token used for averaging
        assert len(self.tokenizer.encode(tok))==3
        with torch.no_grad():
            examples = queries
            hidden_state_list = []
            for batch in list(chunks(examples, self.batch_size)):
                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask)
                hidden_states=output.encoder_last_hidden_state # batch_size x n_tokens x hidden_state_dim
                avg_idx = input_ids==tok_enc
                hidden_states=hidden_states[avg_idx,:].detach().cpu().squeeze().numpy() # get hidden state for gen token
            return(hidden_states)

    def generate(
            self,
            queries,
            decode_method="beam",
            num_beams=5,
            num_generate=5,
            **kwargs,
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            scores = []
            scores_ = None
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_beams,
                    num_return_sequences=num_generate,
                    **kwargs,
                    )

                summaries=out
                scores_=None

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)
                scores.append(scores_)
                return decs

if __name__ == '__main__':
    '''Example of using COMET.'''

    comet = Comet('/home/cgagne/SKAIG-ERC_Repro/pretrained/comet-atomic_2020_BART')
    comet.model.zero_grad()
    context = 'Your partner is mad at you.'
    rel_set = ['xAttr','xIntent','xNeed','xReason','isAfter','HinderedBy','xEffect']

    queries = []
    for rel in rel_set:
        query = "{} {} [GEN]".format(context, rel)
        queries.append(query)

    gens = comet.generate(queries)
    print(gens)

    hidden_state_list = comet.get_hidden(queries)
    print(hidden_state_list[0])
