### how to train the model and finetune it for ellipsis
Clean data first: `clean_data.py`

- Train german LLM: `finetune.py --dataset gcc --model_name <MODELNAME> --pretrained_model <HUGGINGFACE_CHECKPOINT>` only tested on 't5-small' and 't5-base'
- Finetune the german LLM on Tüba or Tiger: `finetune.py --dataset <tiger/tüba> --model_name <MODELNAME> --pretrained_model <PATH/TO/PRETRAINED/LLM>` 
- Evaluate a Model: `test_model.py --checkpoint <MODELNAME>  --corpus <eval/tiger/tuba> --prefix 02sep_TiSm`

In general please have a look at the Bash Scripts (.sh files). These show how the specific models were trained. 