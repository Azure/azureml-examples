# Testing

This test suite heavily borrows from [HF Transformers](https://github.com/huggingface/transformers/), therefore you can refer to the its [testing docs](https://huggingface.co/transformers/testing.html) for in-depth details. In particular wrt writing new tests, as we have access a lot of helper classes and functions, so you can write tests very quickly and not need to reinvent the wheel.

The foundation is `pytest`, which allows you to write normal `pytest` tests, but we also use a lot of unit tests in particular via `TestCasePlus` which extends `unittest` and provides additional rich functionality.

## Running testing

```
make test
```
or:

```
pytest tests
```

Important: the first time you run this it can take some minutes to build all the Megatron cuda kernels and deepspeed kernels if you haven't pre-built the latter.

For various other options please see the doc mentioned at the very top.

You will want to have at least 1 gpu available, best 2 to run the tests.

## CI

The CI setup is documented [here](../.github/workflows/ci.md).
