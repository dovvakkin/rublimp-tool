# rublimp-tool

This is a tool to help you eval your models on https://github.com/RussianNLP/RuBLiMP benchmark

## Usage:

Prepare environment (clone `rublimp` and install python dependencies)

```sh
bash prepare.sh
```

Eval your model one one or all RuBLiMP dataset subsets:

```sh
python rublimp_tool.py eval-model --model 'Qwen/Qwen2.5-0.5B' --subset add_new_suffix --subset add_verb_prefix --output eval.csv
```

Calculate accuracies for one or more evaled models:

```sh
python rublimp_tool.py compare-results --model-type decoder --output accuracies.csv eval.csv
```
