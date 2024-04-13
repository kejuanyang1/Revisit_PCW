
## Get Started
### Manual Installation 
Please first install PyTorch and [apex](https://github.com/NVIDIA/apex), and then install other dependencies by `pip install -r requirements.txt`

## Usage
We provide example scripts for evaluation on classification and CoT tasks.
```shell
bash eval_classification.sh
bash eval_cot.sh
```
Change DATA_DIR in tasks/dataset_loader.py to evaluate on different tasks.
