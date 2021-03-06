# cmr2txt_data_and_eval

This repository contains model outputs for the E2E Challenge and Viggo NLG datasets created for the paper 
[Controllable Meaning Representation to Text Generation: Linearization and Data Augmentation Strategies](https://aclanthology.org/2020.emnlp-main.419/).

# Obtaining validation and test set data.

To access the data unzip the following files:
- e2e_test_model_outputs.zip
- e2e_valid_model_outputs.zip
- viggo_model_outputs.zip

If you run: 

```shell
unzip viggo_model_outputs.zip
unizp e2e_valid_model_outputs.zip
unzip e2e_test_model_outputs.zip
```

this should create the following directory structure:

```shell
model_outputs/{E2E,Viggo}/{BART,bi-gru,transformer}/{valid,test}
```

with each directory containing a jsonl files of model outputs (a jsonl file is a text file where is line is a new json object).
The key to interpretting the json file names is the following:
```{dataset}.{data-partition}.{arch}.{linearization}.{dataaug}.{randomseed}.jsonl```
where:
- `dataset` is either E2E or Viggo
- `data-partition` - validation or test set
- `arch` - BART, transformer, or bi-gru
- `linearization` - the linearization strategy and plan generation used.
    - `rnd` - Random linearization, no plan generation used.
    - `if` - Increasing Frequency linearization, no plan generation used.
    - `fp` - Fixed Position linearization, no plan generation used.
    - `at_bgup` - Alignment Training, plans generated using the Bigram Utterance Planner.
    - `at_nup` - Alignment Training, plans generated using the Neural Utterance Planner.
    - `at_oracle` - Alignment Training, plans generated using a human reference oracle.
- `dataaug` - whether synthetic data was added to the model's training data.
    - `b` indicates only the original training data is used. 
    - `bp` indicates the phrase based  dataugmentation is used to add aditional training examples.
- `randomseed` - each model was trained and evaluated 5 different times using distinct random seeds.


# Data Schema

Each model output example (line in a jsonl file) has the following schema:
- `"mr"` - a dictionary containing the input meaning representation, i.e. dialog act (da) and slot values from which to generate an utterance
-  `"input"` - the meaning representation as it was turned into a flat sequence by the linearization strategy. This is what is fed into the actula seq2seq NLG model.
- `"input_slot_fillers"` - the input sequence with the dialog act (and rating on the Viggo dataset) removed.
- `"references"` - a string containing the human references for this example. Multiple references are delimited by a new line `"\n"` character.  
- `"outputs"` - the beam search outputs of the model.
- `"reranked_beam_output_index"` - the index of outputs that was picked by the beam reranker. This is beam index of the output used for evaluation.



Each output in `"outputs"` list has the following schema:
- `"tokens"` - a list of tokens generated by the model.
- `"pretty"` - the pretty string of the tokens generated by the model. The tokens were detokenized, and any placeholder tokens replaced with their actual values.
- `"tags"` - the tag sequence of the rule based tagger for parsing the actual meaning representation of the generated utterance.
- `"pred_lmr"` - the predicted linearize meaning representation implied by the tag sequence. After human correction, this is used to evaluate semantic error rate and the order acccuracy.
- `"mean_log_prob"` - the mean log probability of the generated tokens.
- `"err"` - the number of semantic errors (without human correct) which were used by the beam reranker.
- `"beam_candidate_num"` - rank of this output in the beam
- `"reranked_beam_candidate_num"` - rank of this output in the reranked beam.

# Iterating over model outputs

To iterate through the outputs of the a file use the following code snippet:

```python
import pathlib
import json

path = pathlib.Path("PATH/TO/JSONL/FILE")

with path.open("r") as fh:
    for line in fh:
        example = json.loads(line)
        output = example["outputs"][example["reranked_beam_output_index"]]
        print("LINEARIZED MR INPUTS")
        print(output["input"])
        print("MODEL GENERATED TOKENS")
        print(output["tokens"])
        print("MODEL GENERATED PRETTY STRING")
        print(output["pretty"])
        print()
```

# Running the evaluation code

To reproduce the metrics in the paper, you can use the following script:
`scripts/evaluate.py`.

To use the script you first need to install the official 
E2E Challenge evaluation repository:
[e2e-metrics](https://github.com/tuetschek/e2e-metrics).

Additionally, make sure numpy and pandas are installed in your Python environment.

To run the evaluation, run:

`python scripts/evaluate.py [--corrections CORRECTIONS] MEASURE_SCORES OUTPUTS_JSONL [OUTPUTS_JSONL] ...`

`evaluate.py` takes an optional argument `--corrections` argument to link to 
the manual corrections of the rule based meaning representation tagger.
The file used for the paper results is lmr.corrections.json`. 

The first positional argument `MEASURE_SCORES` is the path to the e2e-metrics 
repository
evaluation script: `e2e-metrics/measure_scores.py`

The final arguments `OUTPUTS_JSONL` are one or more paths of the output jsonl files. 

For example to reproduce E2E test set scores of the BART model using 
alignment training with the neural utterance planner and without
phrase based data augmentation, run:

```python
python scripts/evaluate.py --corrections lmr.corrections.json e2e-metrics/measure_scores.py model_outputs/E2E/BART/test/E2E.test-og.BART.at_nup.b.*
```
and you should get the following output:

```shell
Read 2220 corrections ...
model_outputs/E2E/BART/test/E2E.test-og.BART.at_nup.b.145480975.jsonl
model_outputs/E2E/BART/test/E2E.test-og.BART.at_nup.b.149691974.jsonl
model_outputs/E2E/BART/test/E2E.test-og.BART.at_nup.b.483242169.jsonl
model_outputs/E2E/BART/test/E2E.test-og.BART.at_nup.b.516253130.jsonl
model_outputs/E2E/BART/test/E2E.test-og.BART.at_nup.b.642689686.jsonl
                                        path    BLEU     NIST  METEOR  ROUGE_L   CIDEr  missing  incorrect  added   all   SER (%)  Order (Acc. %)   Perf (%)
0  E2E.test-og.BART.at_nup.b.145480975.jsonl  66.220  8.65060   45.56    69.13  2.3006      0.0        0.0   11.0  11.0  0.252757       98.253968  98.253968
1  E2E.test-og.BART.at_nup.b.149691974.jsonl  66.460  8.68670   45.48    69.13  2.3125      0.0        0.0   13.0  13.0  0.298713       97.936508  97.936508
2  E2E.test-og.BART.at_nup.b.483242169.jsonl  66.390  8.66280   45.45    69.31  2.3171      0.0       10.0    4.0  14.0  0.321691       97.777778  97.777778
3  E2E.test-og.BART.at_nup.b.516253130.jsonl  66.730  8.67710   45.47    69.07  2.3193      0.0        0.0    3.0   3.0  0.068934       99.523810  99.523810
4  E2E.test-og.BART.at_nup.b.642689686.jsonl  66.980  8.73220   45.59    69.51  2.3230      0.0        0.0    2.0   2.0  0.045956       99.682540  99.682540
0                                       mean  66.556  8.68188   45.51    69.23  2.3145      0.0        2.0    6.6   8.6  0.197610       98.634921  98.634921

```

