# -*- coding: utf-8 -*-

from data.batch import Batch
from config.params import Params
from data.shared_dataset import SharedDataset
from model.model import Model
import os.path
import torch
import json
from datetime import date
from tqdm import tqdm
import sys

input_file=sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
filename="/home/xhd160/perin-parsing/weights/english/base_ucca.h5"
output_path=input_file+".mrp"
lines=open(input_file,"r",encoding="utf-8").readlines()
lines=[line[:-2] for line in lines]
print(output_path)
def load_checkpoint(filename, device):
    state_dict = torch.load(filename, map_location=device)
    args = Params().load_state_dict(state_dict["args"])

    dataset = SharedDataset(args)
    dataset.load_state_dict(args, state_dict["dataset"])

    model = Model(dataset, args, initialize=False).to(device).eval()
    model.load_state_dict(state_dict["model"])
    
    return model, dataset, args

model, dataset, args = load_checkpoint(filename, device)

def parse_mid(input, model, dataset, args, language, **kwargs):
    # preprocess
    batches = dataset.load_sentences(input, args, "ucca", language)
    output = batches.dataset.datasets[dataset.framework_to_id[("ucca", language)]].data
    output = list(output.values())
    
    for i, batch in tqdm(enumerate(batches)):
        # parse and postprocess
        with torch.no_grad():
            prediction = model(Batch.to(batch, device), inference=True, **kwargs)[("ucca", language)][0]

        for key, value in prediction.items():
            output[i][key] = value

        # clean the output
        output[i]["input"] = output[i]["sentence"]
        output[i] = {k: v for k, v in output[i].items() if k in {"id", "input", "nodes", "edges", "tops"}}
        output[i]["framework"] = "ucca"
        
    return output

def parse(input, model, dataset, args, language, **kwargs):
    # preprocess
    outputs=[]
    for i in tqdm(range(int(len(input)/100))):
        #try:
            output=parse_mid(input[i*100:(i+1)*100], model, dataset, args, language, **kwargs)
            outputs=outputs+output
        #except:
        #    print("problem with range: "+str(100*i))
    #try:
    #   output=parse_mid(input[(i+1):], model, dataset, args, language, **kwargs)
    #   outputs=outputs+output
    #except:
    #       pass
    return outputs
# Save the parsed graph into json-like MRP format.
#
def save(output, path):
    with open(path, "w", encoding="utf8") as f:
        for sentence in output:
            json.dump(sentence, f, ensure_ascii=False)
            f.write("\n")


sentences = lines
language = "eng"  # available languages: {"eng", "zho"}

prediction = parse(sentences, model, dataset, args, language, approximate_anchors=False)
save(prediction, output_path)
