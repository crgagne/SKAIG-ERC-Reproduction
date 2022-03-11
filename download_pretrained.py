import warnings
from os import path
from transformers import RobertaModel

def download_and_save(modelname, directory, sentence_transformer=False, cross_encoder=False):

    save_path = path.join(directory, modelname)
    model = RobertaModel.from_pretrained(modelname)
    model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    tokenizer.save_pretrained(save_path)

def main():

    save_dir = path.join(path.dirname(__file__), "pretrained")
    download_and_save("roberta-large", save_dir)

if __name__ == '__main__':
    main()
