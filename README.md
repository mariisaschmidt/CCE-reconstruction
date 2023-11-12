# Clausal Coordinate Ellipsis

- [Clausal Coordinate Ellipsis](#clausal-coordinate-ellipsis)
  - [Marisa's local conda-environment](#marisas-local-conda-environment)
    - [Packages](#packages)
  - [Marisa's VM](#marisas-vm)
  - [First Steps](#first-steps)
  
## Marisa's local conda-environment
conda activate cce

### Packages
matplotlib                3.8.0           py310hb6292c7_2    conda-forge
numpy                     1.26.0          py310he2cad68_0    conda-forge
pip                       23.2.1             pyhd8ed1ab_0    conda-forge
python                    3.10.0          h43b31ca_3_cpython    conda-forge
pytorch                   2.0.0           cpu_generic_py310h9b0b4f9_3    conda-forge
tokenizers                0.14.1          py310h4a533d7_2    conda-forge
tqdm                      4.43.0                     py_0    conda-forge
transformers              4.34.1             pyhd8ed1ab_0    conda-forge
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