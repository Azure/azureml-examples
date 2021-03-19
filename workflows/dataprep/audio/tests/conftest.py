from os.path import join
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def tmpdir_audio_files(tmpdir_factory):
    input_dir = str(tmpdir_factory.mktemp("audio"))
    sf.write(join(input_dir, "sample_1.wav"), np.random.normal(0, 1, 22050), 22050)
    sf.write(join(input_dir, "sample_2.wav"), np.random.normal(0, 1, 22050), 22050)

    print(list(Path(input_dir).rglob("*")))

    return input_dir
