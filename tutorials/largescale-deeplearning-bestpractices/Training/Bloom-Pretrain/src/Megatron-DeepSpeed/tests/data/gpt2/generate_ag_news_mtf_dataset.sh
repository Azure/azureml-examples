python -c "from datasets import load_dataset; load_dataset('TimeRobber/ag_news_classify_question_first_100', split='train').to_json('ag_news_classify_question_first_100.jsonl')"

python tools/preprocess_data.py \
    --input ag_news_classify_question_first_100.jsonl \
    --output-prefix tests/data/gpt2/ag_news_prompt \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path bigscience/tokenizer \
    --append-eod \
    --workers 8

python tools/preprocess_data.py \
    --input ag_news_classify_question_first_100.jsonl \
    --output-prefix tests/data/gpt2/ag_news_prompt \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path bigscience/tokenizer \
    --workers 8

rm ag_news_classify_question_first_100.jsonl
