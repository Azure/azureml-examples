"""
test_audio.py
"""
import filecmp
import os
import tempfile
import uuid

import numpy as np
import pytest
import soundfile as sf
from src.audio import (
    compress,
    denoise,
    denoise_whole_audio,
    find_noise,
)

t = np.linspace(0, 4.0, 4000)  # 1 second
noise = np.random.normal(0, 0.01, 2000)
sig = 2 * np.sin(2 * np.pi * 5 * t[0:2000])
NOISE_SIG = np.asarray([noise, sig]).reshape(-1)
RATE = 1000  # Fake sampling rate lower than default 44.1 kHz


@pytest.mark.parametrize(
    "signal, expected_signal, clip_flag",
    [(NOISE_SIG, NOISE_SIG[:1000], True), (sig, np.zeros(1000), False)],
)
def test_find_noise(signal, expected_signal, clip_flag):
    """
    test find_noise function

    Parameters
    -------
    signal : np.ndarray
        [input signal]
    expected_signal : np.ndarray
        [expected output]
    clip_flag : book
        [flag for found/not found noise clip]

    """
    # Arrange
    noise_len = 1

    # Act
    noise_clip, found_clip = find_noise(signal, sample_rate=1000, noise_len=noise_len)

    # Assert
    assert found_clip == clip_flag
    np.testing.assert_array_almost_equal(noise_clip, expected_signal, decimal=1)


@pytest.mark.parametrize("input_sig, rate", [(NOISE_SIG, RATE)])
def test_denoise_whole_audio(input_sig, rate):
    """
    test denoise_whole_audio

    Parameters
    -------
    input_sig : np.ndarray
        [The audio signal]
    rate : int
        [Sampling rate in Hz]
    """
    # Act
    sig_filt, rate_filt = denoise_whole_audio(input_sig, rate)

    # Assert
    assert rate == rate_filt
    assert len(input_sig) == len(sig_filt)
    assert (
        type(sig_filt[0]) in [float, np.float32, np.float64]
    ) is True  # Check value type
    assert (True in np.isnan(sig_filt)) is False  # No NaN values
    assert (len(np.unique(sig_filt)) > 2) is True  # Values are not all zeros


@pytest.mark.parametrize(
    "input_audio, sample_rate",
    [(np.random.normal(0, 1, 22050), 22050), (np.random.normal(0, 1, 44100), 44100)],
)
def test_compress(input_audio, sample_rate):
    """Tests `compress`

    Parameters
    ----------
    input_audio : np.ndarray
        Parameter to test
    sample_rate : int
        Parameter to test
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio_filepath = os.path.join(temp_dir, str(uuid.uuid4()))
        sf.write(input_audio_filepath, input_audio, sample_rate)
        output_audio_filepath = os.path.join(temp_dir, "compressed_audio.wav")

        compress(input_audio_filepath, output_audio_filepath)

        assert os.path.exists(output_audio_filepath)
        assert not filecmp.cmp(input_audio, output_audio_filepath, shallow=False)
