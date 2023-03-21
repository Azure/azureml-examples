# How to run lm-eval on Megatron-DeepSpeed checkpoint using the original setup

This particular setup uses the normal deepspeed checkpoint and requires no conversion to Megatron-LM.

This doc assumes usage on JZ, so some peculiar requirements in places. Ignore these if you're not running this on JZ.

## Prerequisites

1. Install software

On login console with external network

Get lm-eval harness (https://github.com/EleutherAI/lm-evaluation-harness) and `best-download==0.0.7` needed to download some tasks.
```
start-prod
pip install best-download==0.0.7
pip install git+https://github.com/EleutherAI/lm-evaluation-harness
```

2. Pre-download needed datasets

some symlinks due to lm-harness' issues with relative position of data
```
mkdir data
ln -s `pwd`/data tasks/eval_harness/data
```
Also make sure `data` is not on one of the limited paritions like WORKSF.

Then install datasets for the tasks:
```
python ./tasks/eval_harness/download.py --task_list
arc_challenge,arc_easy,boolq,copa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc
```
and make sure that `export HF_DATASETS_OFFLINE=1`

If there are things like custom tokenizers, pre-download those too, e.g.:

```
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bigscience/oscar_13_languages_alpha_weight')"
```
and make sure that `export TRANSFORMERS_OFFLINE=1` is in the script.
You know there is a custom tokenizer if the training script had something like:

```
--tokenizer-type PretrainedFromHF \
 --tokenizer-name-or-path bigscience/oscar_13_languages_alpha_weight \
```

3. Prepare the slurm script

Prepare the run script, replace `variant` with a unique identifier for the current eval so that multiple evals could run in parallel and not all log into the same `results.json` file. so, e.g., `tr9c-1B3-swiglu`

```
cp examples/run_evalharness_deepspeed.slurm run_evalharness-variant.slurm
```

now edit `run_evalharness-variant.slurm`


Note that the eval code knows to pull the original training args from the checkpoint, so we don't need to pass any of those. And we just need to setup the evaluation args.

1. Edit:

```
PP_SIZE=1
TP_SIZE=1
```
to match the eval topology. If the model fits into 1 gpu, then there is nothing to change.

The eval script will automatically reshape the model if it was of a different topology.


2. Adjust the following to fit the chosen GPU. As of last check for 1.3B model the settings are one of:
```
EVAL_MICRO_BATCH_SIZE=6  # 16GB GPU 1.3B model
EVAL_MICRO_BATCH_SIZE=12 # 32GB GPU 1.3B model
```

If you get OOM lower it further.

3. If not using the Deepspeed path, disable it by removing:

```
    --deepspeed \
    --deepspeed_config ds_config.json \
```

If you didn't disable it and the program crashed on checkpoint loading unable to find some key, disable deepspeed as explained above.

4. Additional flags

- To reduce the amount of iterations for stderr estimation, use e.g. `--bootstrap_iters 2`. This saves 1-2 minutes per dataset.
- To print intermediate results when running multiple tasks use `--intermed_results`.
- To reduce the bubble when setting PP use the flag `--micro_bs_multiplier`. Reducing `--micro-batch-size` may be needed when increasing the multiplier. 
    - Running the 176B model with PP=8, `--micro_bs_multiplier 8` & `--micro-batch-size 4` produced the fastest results for PiQA on 1 node in 2min18s.

## Eval

Currently it takes 2-3 hours to run on 32GB for 1.3B model, 6-7h for 16GB GPU, so a 20h slurm job should be enough.

When ready, launch:
```
sbatch ./run_evalharness-variant.slurm
```

To monitor progress:
```
tail -f tail -f $VARIANT-eval-harness.log
```
where the variant is what you set `$VARIANT` to in the slurm script.

The template is set up for 16GB gpu since they are easier to get by. If you change to 32GB, adjust:
```
#SBATCH --constraint=v100-32g
...
EVAL_MICRO_BATCH_SIZE=12 # 32GB GPU 1.3B model
```


Note that the original ETA at the start of the run can be 10x too longer than the actual outcome. For example it may suggest 18 hours but will complete in 2 hours.


## Short eval

if you just want to quickly test that everything can run to the end, edit `tasks/eval_harness/evaluate.py`,  e.g. to run only 10 batches:
```
- results = evaluator.evaluate(adaptor, task_dict, False, 0, None)
+ results = evaluator.evaluate(adaptor, task_dict, False, 0, 10)
```

(XXX: could be a cmd line option so that code won't need to be modified)


## Import into spreadsheet

https://docs.google.com/spreadsheets/d/1CI8Q9RCblLRzUOPJ6ViqBmo284-8ojluQ-CmaEuhuv0/edit?usp=sharing

Note that the spreadsheet format is quite different, so use this script:
```
./tasks/eval_harness/report-to-csv.py results.json
```
to reformat the json results into csv while changing its shape to match the spreadsheet format

Since some records might be missing or extraneous here is the best way to do it:

1. copy the data from first 2 columns to some place under the main spreadsheet

2. put the pointer to the 3rd column next to where the 2 first columns were copied.

3. import `results.csv` using file-> import -> file ->

Import location: Replace data at selected cell

4. Now it should be easy to align the new records with the old ones - delete irrelevant records and Insert->Cells where data is missing until the first 2 columns match

5. now create 2 cols in the main table on top and now it should be safe to Copy-n-Paste the 2-col data range, without the task/metrics columns into the newly created space.
