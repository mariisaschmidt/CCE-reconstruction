# Clausal Coordinate Ellipsis

- [Clausal Coordinate Ellipsis](#clausal-coordinate-ellipsis)
  - [Marisa's local conda-environment](#marisas-local-conda-environment)
    - [Packages](#packages)
  - [Marisa's VM](#marisas-vm)
  - [First Steps](#first-steps)
  - [Final](#final)
  
## Marisa's local conda-environment
conda activate cce

### Packages     
pip    
python == 3.10
transformers 
datasets  
tokenizers    
pytorch
matplotlib
numpy          
tqdm       
argparse

## Marisa's VM
ssh marisa@141.26.157.90 (via Uni VPN)

## First Steps
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- uses Encoder-Decoder-Model
- create translation pairs file with read_dataset.py in Cloud/Uni/EllipsenProjekt 
- implementation and results can be found at /sequence2sequence

https://huggingface.co/docs/transformers/tasks/translation
- install Transformers library
- also sequence to sequence

## Final 
see german-common-crawl folder