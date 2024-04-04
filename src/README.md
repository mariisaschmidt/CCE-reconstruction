### how to train the model and finetune it for ellipsis
- Train german LLM: `finetune.py --dataset gcc --model_name <MODELNAME> --pretrained_model <HUGGINGFACE_CHECKPOINT>` only tested on 't5-small'
- Finetune the german LLM on Tüba or Tiger: `finetune.py --dataset <tiger/tüba> --model_name <MODELNAME> --pretrained_model <PATH/TO/PRETRAINED/LLM>` 