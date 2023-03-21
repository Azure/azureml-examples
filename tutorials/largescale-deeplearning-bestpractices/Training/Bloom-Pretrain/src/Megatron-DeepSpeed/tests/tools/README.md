# Test suite tools

# Make tiny tokenizer files

currently for gpt2 run:
```
./shrink-tokenizer.py
```

and then we have tiny vocab and merge files under the generated dir `tiny` to add to repo under `data/gpt2`.

```
cp tiny/merges.txt ../data/gpt2/gpt2-tiny-merges.txt
cp tiny/vocab.json ../data/gpt2/gpt2-tiny-vocab.json
```

Note, the tiny vocab was set to 5000 items after experimenting with the resulting index files size. Using a tiny vocab of 500 (and adjusted merge entries) proved to generate very large index files, so it actually ends up costing more in final file size. 5000 proved to generate an almost identical index files as with the original 50k vocab size.


# Make tiny pre-processed index

to be used in test training

```
./openwebtext-to-jsonl.py
```

generates:

```
openwebtext-10000.jsonl
```

we don't want to store jsonl in repo, to keep the size small, so it's a temp file.

Now we pre-process it:

```
cd ../..
input=tests/tools/openwebtext-1000.jsonl
python tools/preprocess_data.py \
    --input $input \
    --output-prefix tests/data/gpt2/meg-gpt2-openwebtext \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file tests/data/gpt2/gpt2-tiny-merges.txt \
    --vocab tests/data/gpt2/gpt2-tiny-vocab.json \
    --append-eod \
    --workers 6
```

and voila we now have:
```
ls -sh1 tests/data/gpt2/meg-gpt2-openwebtext*
2.6M tests/data/gpt2/meg-gpt2-openwebtext_text_document.bin
 20K tests/data/gpt2/meg-gpt2-openwebtext_text_document.idx
```
which we can now commit and use in tests.
