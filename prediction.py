import torch
from tqdm import tqdm, trange
from util.data_process import makeDataLoader
from transformers import AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


max_seq_length = 64
id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}

def predict(model, df_for_test, batch_size):

    model = model.to(device)
    test_dataloader = makeDataLoader(df_for_test, 2, batch_size, False)
    bert_tokenizer = AutoTokenizer.from_pretrained('./model/bert_tokenizer')
    prediction_results = []
    for batch in tqdm(test_dataloader):
        model.eval()
        
        b_text, b_imgs = batch
        b_inputs = bert_tokenizer(list(b_text), truncation=True, max_length=max_seq_length, return_tensors="pt", padding='max_length')

        # b_labels = b_labels.to(device)
        b_inputs = b_inputs.to(device)
        b_imgs = b_imgs.to(device)
        
        with torch.no_grad():
            b_logits = model(b_inputs, b_imgs)
            b_logits = b_logits.detach().cpu()
            
        prediction_results += torch.argmax(b_logits, dim=-1).tolist()
        
        
    prediction_labels = [id_to_label[p] for p in prediction_results]

    return prediction_labels



