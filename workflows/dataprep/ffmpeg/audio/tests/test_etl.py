"""
Test etl.py
"""
import sys
import tempfile
from os.path import join
from pathlib import Path

import pytest
from mlops.steps.etl import init, run
from src.aml import get_logger

log = get_logger(__name__)


@pytest.mark.parametrize(
    "sample_rate, transform_order",
    [
        (0, ""),
        (0, "compress"),
        (0, "denoise"),
        (0, "compress, denoise"),
        (0, "denoise,compress"),
        (16000, ""),
    ],
)
def test_etl(tmpdir_audio_files: str, sample_rate: int, transform_order: str):
    """Test init() and run() for etl.py

    Due to the way ParallelRunStep, it is necessary to run `init` then
    `run` so the global variables are set from the argparse

    Parameters
    ----------
    sample_rate : int
        Parameter to test
    transform_order : str
        Parameter to test
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_dir = str(tempdir)

        sys.argv[1:] = [
            "--base-dir",
            "",
            "--input-dir",
            tmpdir_audio_files,
            "--output-dir",
            output_dir,
            "--overwrite",
            str(True),
            "--sample-rate",
            str(sample_rate),
            "--transform-order",
            transform_order,
        ]

        init()

        audio_filenames = [
            join(tmpdir_audio_files, Path(f).name)
            for f in Path(tmpdir_audio_files).glob("*.wav")
        ]
        run(audio_filenames)

        assert len(audio_filenames) > 0

        # Directory not empty
        for audio_filename in audio_filenames:
            audio_filename = Path(audio_filename).name
            log.info(join(output_dir, audio_filename))
            assert Path(join(output_dir, audio_filename)).exists()
