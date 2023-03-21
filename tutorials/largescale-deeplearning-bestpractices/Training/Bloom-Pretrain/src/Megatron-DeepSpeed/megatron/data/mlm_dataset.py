"""Non-Causal Mask Language Model Finetune Style dataset."""

import numpy as np
import torch

from megatron import print_rank_0, get_tokenizer, get_args
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples, get_split_by_range_
from megatron.data.dataset_utils import get_train_valid_test_split_, get_indexed_dataset_
from megatron.data.gpt_dataset import GPTDataset


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    sequence_length,
                                    noise_density,
                                    mean_noise_span_length,
                                    seed,
                                    skip_warmup
                                    ):
    assert noise_density is not None
    assert mean_noise_span_length is not None

    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(
            data_prefix=data_prefix[0],
            data_impl=data_impl,
            splits_string=splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            sequence_length=sequence_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            seed=seed,
            skip_warmup=skip_warmup
        )
    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            data_prefix=prefixes[i],
            data_impl=data_impl,
            splits_string=splits_string,
            train_valid_test_num_samples=datasets_train_valid_test_num_samples[i],
            sequence_length=sequence_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            seed=seed,
            skip_warmup=skip_warmup
        )
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

        # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)

def build_dataset_group(
    dataset_group_name,
    paths,
    weights,
    splits,
    data_impl,
    train_valid_test_num_samples,
    seq_length,
    noise_density,
    mean_noise_span_length,
    seed,
    skip_warmup,
    train_valid_test
):
    '''
    Build a single dataset group corresponding to Option 2 of data loading see arguments.py
    a dataset group is passed on the following form
    GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT2 START:END PATH2
    or alternatively
    GIVEN_NAME PATH1    # for a single dataset to be used fully
    '''

    assert train_valid_test in ["train","valid","test"]

    # Single dataset.
    if len(paths) == 1:
        dataset = _build_single_datasets(
            data_prefix=paths[0],
            range_string=splits[0],
            data_impl=data_impl,
            train_valid_test_num_samples=train_valid_test_num_samples,
            sequence_length=seq_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            seed=seed,
            skip_warmup=skip_warmup,
            dataset_group_name=dataset_group_name,
            train_valid_test=train_valid_test)
        return dataset
    # Blending dataset.
    else:

        data_prefix = []
        # data_prefix is on the shape:
        # ["WEIGHT1", "PATH1", "WEIGHT2", "PATH2", "WEIGHT3", "PATH3"]
        for w,p in zip(weights, paths):
            data_prefix += [w,p]

        output = get_datasets_weights_and_num_samples(data_prefix,
                                                    train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_single_datasets(
                data_prefix=prefixes[i],
                range_string=splits[i],
                data_impl=data_impl,
                train_valid_test_num_samples=datasets_train_valid_test_num_samples[i],
                sequence_length=seq_length,
                noise_density=noise_density,
                mean_noise_span_length=mean_noise_span_length,
                seed=seed,
                skip_warmup=skip_warmup,
                dataset_group_name=dataset_group_name,
                train_valid_test=train_valid_test
            )
            datasets.append(ds)
        all_datasets = BlendableDataset(datasets, weights)

        return all_datasets

def _build_single_datasets(
        data_prefix,
        range_string,
        data_impl,
        train_valid_test_num_samples,
        sequence_length,
        noise_density,
        mean_noise_span_length,
        seed,
        skip_warmup,
        dataset_group_name,
        train_valid_test):
    """Build a single dataset"""

    assert train_valid_test in ["train","valid","test"]
    index = ["train","valid","test"].index(train_valid_test)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    # this corresponds to option2 for data loading on the form
    # WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT3 START:END PATH3
    # splits here is an array of size 2  [start_index, end_index]
    splits = get_split_by_range_(range_string=range_string, size=total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    print_rank_0('    {}:'.format(dataset_group_name))
    print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[0], splits[1],
                                        splits[1] - splits[0]))

    def build_dataset(name):
        dataset = None
        if splits[1] > splits[0]:
            documents = np.arange(start=splits[0], stop=splits[1],
                                  step=1, dtype=np.int32)
            dataset = MLMDataset(
                indexed_dataset=indexed_dataset,
                documents=documents,
                noise_density=noise_density,
                mean_noise_span_length=mean_noise_span_length,
                name=name,
                data_prefix=data_prefix,
                sequence_length=sequence_length,
                num_samples=train_valid_test_num_samples[index],
                seed=seed,
            )
        return dataset

    dataset = build_dataset(dataset_group_name)

    return dataset

def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     sequence_length,
                                     noise_density,
                                     mean_noise_span_length,
                                     seed,
                                     skip_warmup):
    """Build train, valid, and test datasets."""


    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
        start_index = indexed_dataset.doc_idx[splits[index]]
        end_index = indexed_dataset.doc_idx[splits[index + 1]]
        print_rank_0('     sentence indices in [{}, {}) total of {} '
                     'sentences'.format(start_index, end_index,
                                        end_index - start_index))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            # Build the dataset accordingly.
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = MLMDataset(
                    indexed_dataset=indexed_dataset,
                    documents=documents,
                    noise_density=noise_density,
                    mean_noise_span_length=mean_noise_span_length,
                    name=name,
                    data_prefix=data_prefix,
                    sequence_length=sequence_length,
                    num_samples=train_valid_test_num_samples[index],
                    seed=seed,
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


class MLMDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        indexed_dataset,
        documents,
        data_prefix,
        sequence_length,
        num_samples,
        seed,
        noise_density=0.15,
        mean_noise_span_length=3
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.sequence_length = sequence_length

        # Dataset.
        self.indexed_dataset = indexed_dataset

        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
        # To ensure that the input length is `sequence_length`, we need to increase the maximum length
        # according to `noise_density` and `mean_noise_span_length`. We can also define the label length accordingly.
        number_of_raw_tokens, inputs_length, targets_length, num_noise_spans = compute_input_and_target_lengths(
            sequence_length=self.sequence_length,
            noise_density=self.noise_density,
            mean_noise_span_length=self.mean_noise_span_length
        )
        self.inputs_length = inputs_length
        # In order to compute loss, we need an extra token at the end.
        self.number_of_raw_tokens = number_of_raw_tokens + 1
        self.targets_length = targets_length + 1
        self.num_noise_spans = num_noise_spans

        # Build the samples mapping.
        self._gpt_dataset = GPTDataset(
            name=self.name,
            data_prefix=data_prefix,
            documents=documents,
            indexed_dataset=self.indexed_dataset,
            num_samples=num_samples,
            # -1 because GPTDataset will return `seq_length + 1` sequences.
            seq_length=self.number_of_raw_tokens - 1,
            seed=seed
        )

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.sep_id = tokenizer.sep
        self.sentinel_token_ids = tokenizer.additional_special_tokens_ids
        assert self.sep_id is not None, "MLM dataset requires tokenizer to have a <sep> token"
        assert len(self.sentinel_token_ids) > 0, "Provide the argument --vocab-extra-ids 100 to the script"
        assert len(self.sentinel_token_ids) >= self.num_noise_spans, "Not enough sentinel tokens, please add more"

        args = get_args()
        # TODO @thomasw21 check once we merge t5
        assert self.inputs_length + self.targets_length == args.seq_length + 1

    def __len__(self):
        return len(self._gpt_dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError

        sample = self._gpt_dataset[idx]["text"]

        return build_training_sample(
            sample=sample,
            inputs_length=self.inputs_length,
            targets_length=self.targets_length,
            num_noise_spans=self.num_noise_spans,
            sep_id=self.sep_id,
            all_sentinel_token_ids=self.sentinel_token_ids,
        )


def build_training_sample(
    sample,
    inputs_length,
    targets_length,
    num_noise_spans,
    sep_id,
    all_sentinel_token_ids,
):
    """Build training sample.

    Arguments:
        sample: int32 tensor
        inputs_length: integer
        targets_length: integer
        num_noise_spans: integer
        sep_id: integer
        all_sentinel_token_ids: List[int]
    Returns:
        Dict with following keys:
            - `input_tokens`: int32 tensor with as length input_length,
            - `target_tokens`: int32 tensor with as length targets_length + 1,
    """

    spans_start, mask_indices = random_spans_noise_mask(
        inputs_length=inputs_length,
        targets_length=targets_length,
        num_noise_spans=num_noise_spans,
    )
    spans_end = np.concatenate([
        spans_start[1:], np.full((1,), len(sample), dtype=np.int32)]
    )

    sentinel_token_ids = all_sentinel_token_ids[:num_noise_spans]

    input_token_ids = np.concatenate(
        [
            elt
            for start, end, sentinel_token in zip(spans_start[::2], spans_end[::2], sentinel_token_ids)
            for elt in [sample[start: end], np.full((1,), sentinel_token, dtype=np.int32)]
        ] +
        [np.full((1,), sep_id, dtype=np.int32)]
    )
    target_token_ids = np.concatenate(
        [
            elt
            for start, end, sentinel_token in zip(spans_start[1::2], spans_end[1::2], sentinel_token_ids)
            for elt in [np.full((1,), sentinel_token, dtype=np.int32), sample[start: end]]
        ] +
        [np.full((1,), sep_id, dtype=np.int32)]
    )

    return {
        'input_tokens': input_token_ids,
        'target_tokens': target_token_ids
    }


def compute_input_and_target_lengths(sequence_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have SEP appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(_tokens_length):
        num_noise_tokens = int(round(_tokens_length * noise_density))
        num_nonnoise_tokens = _tokens_length - num_noise_tokens
        _num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans and one SEP token.
        _input_length = num_nonnoise_tokens + _num_noise_spans + 1
        _output_length = num_noise_tokens + _num_noise_spans + 1
        return _input_length, _output_length, _num_noise_spans

    tokens_length = sequence_length
    inputs_length, targets_length, num_noise_spans = _tokens_length_to_inputs_length_targets_length(tokens_length)
    while inputs_length + targets_length > sequence_length:
        tokens_length -= 1
        inputs_length, targets_length, num_noise_spans = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # tokens_length is the number of raw tokens we need to get
    # inputs_length will be the input
    # targets_length will be the target
    # num_noise_spans is the number of spans we have to replace
    return tokens_length, inputs_length, targets_length, num_noise_spans


def random_spans_noise_mask(
    inputs_length,
    targets_length,
    num_noise_spans,
):

    """This function is inspired from `random_spans_noise_mask <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    Spans alternate between non-noise and noise, beginning with non-noise.
    Args:
        inputs_length: int32 scalar
        targets_length: int32 scalar
        num_noise_spans: int32 scalar
    Returns:
        a int8 tensor with shape [num_noise_spans]
        a boolean tensor with shape [length]
    """
    # # pick the lengths of the noise spans and the non-noise spans
    num_noise_tokens = targets_length - num_noise_spans - 1
    num_nonnoise_tokens = inputs_length - num_noise_spans - 1
    number_of_raw_tokens = num_noise_tokens + num_nonnoise_tokens

    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        # TODO @thomasw21 handle random state correctly, ie synchronized across TP.
        #   we might not care as get_batch_pipe broadcasts data to all devices.
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]], constant_values=0)
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.concatenate([np.full((1,), 0, dtype=np.int32), np.cumsum(interleaved_span_lengths)[:-1]])
    span_start_indicator = np.zeros((number_of_raw_tokens,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return span_starts, is_noise
