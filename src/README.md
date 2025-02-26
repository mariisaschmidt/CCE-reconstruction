### How to train the model and finetune it for ellipsis:
- Train german-to-german LLM: `finetune.py --dataset gcc --model_name <MODELNAME> --pretrained_model <HUGGINGFACE_CHECKPOINT>` 
  - (tested with 't5-small' and 't5-base')
- Finetune the german LLM on ellipsis data: `finetune.py --dataset <tiger/tüba> --model_name <MODELNAME> --pretrained_model <PATH/TO/PRETRAINED/LLM>` 
  - have a look at the scripts in the scripts-directory to see possible options
- Or use some script from the scripts-directory