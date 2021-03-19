"""
Audio preprocessing module
"""
import shutil
import subprocess
import tempfile
import uuid
from os.path import join
from typing import Tuple

import librosa
import noisereduce as nr
import numpy as np
from scipy.io import wavfile

from .aml import get_logger
from .common import reuse

log = get_logger(__name__)


def find_noise(
    audio: np.ndarray, sample_rate: int, noise_len: int = 1
) -> Tuple[np.ndarray, bool]:
    """
    Automatic noise detection

    Args:
        audio: audio track
        sample_rate: audio sample rate
        noise_len: length of noise clip (in seconds). default to 1
    ---

    Return:
        noise_clip, found_clip: clip with noise and bool_flag. if the noise clip is not found, a np.zeros array is
        returned
    """

    # transform amplitude to dB for noise detection
    audio_db = librosa.amplitude_to_db(audio, ref=np.max)
    found_clip = False
    noise_clip = np.zeros(noise_len * sample_rate)
    for i in range(len(audio_db)):
        clip = audio_db[i : i + noise_len * sample_rate]
        if not found_clip:
            # -35dB is noise
            if np.max(clip) <= -35:
                noise_clip = audio[i : i + noise_len * sample_rate]
                found_clip = True

    # check noise_clip dimension:
    if len(noise_clip) == 1:  # if last audio sample has amplitude < -35dB:
        noise_clip = np.zeros(noise_len * sample_rate)
        found_clip = False

    return noise_clip, found_clip


def denoise_whole_audio(sig: np.ndarray, rate: int) -> Tuple[np.ndarray, int]:
    """Run denoising across all of the audio

    Args:
        sig (np.ndarray): The audio signal
        rate (int): Sampling rate (Hz)

    Returns:
        sig_filt (np.ndarray): The filtered audio signal
        rate (int): Sampling rate (Hz)
    """

    # Check if audio is too long (will lead to MemoryError)
    # Note: Try/Except for MemoryError does not seem to work
    is_over_30_min = len(sig) / rate / 60 > 30

    if is_over_30_min is True:
        # If signal is so big that it can't be processed all at once
        split_a = int(len(sig) / 4)
        split_b = int(len(sig) / 4) * 2
        split_c = int(len(sig) / 4) * 3

        def denoise_partial_signal(
            sig_to_split: np.ndarray, split_point_1: int, split_point_2: int, rate: int
        ) -> np.ndarray:
            """Temporary function to denoise part of a signal

            Args:
                sig_to_split (np.ndarray): The audio signal to split
                split_point_1 (int): Index point to begin split
                split_point_2 (int): Index point to end split
                rate (int): Sampling rate of the signal

            Returns:
                sig_temp (np.ndarray): The denoised partial signal
            """

            # Cast values to float32 if currently at float64
            if isinstance(sig_to_split[0], np.float64):
                sig_to_split = sig_to_split.astype(np.float32)

            split_sig = sig_to_split[split_point_1:split_point_2]
            noise_clip, _ = find_noise(split_sig, rate)

            sig_temp = nr.reduce_noise(
                audio_clip=split_sig, noise_clip=noise_clip, verbose=False
            )
            sig_temp = np.asarray(sig_temp).astype(np.float32)
            sig_temp /= np.max(np.abs(sig_temp), axis=0)  # rescale audio
            return sig_temp

        # Parts
        sig_filt1 = denoise_partial_signal(sig, 0, split_a, rate)
        sig_filt2 = denoise_partial_signal(sig, split_a, split_b, rate)
        sig_filt3 = denoise_partial_signal(sig, split_b, split_c, rate)
        sig_filt4 = denoise_partial_signal(sig, split_c, len(sig), rate)

        # Combine
        sig_filt = np.concatenate([sig_filt1, sig_filt2, sig_filt3, sig_filt4])
        print("Signal processed as parts, then concatenated post-processing")
    else:
        # Detect noise clip
        noise_clip, _ = find_noise(sig, rate)

        sig_filt = nr.reduce_noise(audio_clip=sig, noise_clip=noise_clip, verbose=False)
        sig_filt = np.asarray(sig_filt).astype(np.float32)
        sig_filt /= np.max(np.abs(sig_filt), axis=0)  # rescale audio

    return sig_filt, rate


@reuse
def denoise(
    input_audio_filepath: str,
    output_audio_filepath: str,
    overwrite: bool = False,
):
    """Wrapper function around denoise_whole_audio to match signature required for reuse

    Parameters
    ----------
    input_audio_filepath : str
        Input filepath to the audio file to denoise
    output_audio_filepath : str
        Output filepath to the audio file that is denoised
    overwrite : bool
        True if output_audio_filepath should be overwritten
    """
    # pylint: disable=unused-argument
    # overwrite is used by the decorator reuse.
    # It is kept in this function signature to reflect that it is a valid parameter to pass in.
    sig, sample_rate = librosa.load(input_audio_filepath, sr=None)
    audio_clip, sample_rate = denoise_whole_audio(sig, sample_rate)
    wavfile.write(output_audio_filepath, sample_rate, audio_clip)


@reuse
def compress(input_filepath: str, output_filepath: str, overwrite: bool = True) -> None:
    """Runs dynamic range compression using ffmpeg

    Parameters
    ----------
    input_filepath : str
        Filepath of the wav file to apply dynamic range compression
    output_filepath : str
        Name of output file after running dynamic range compression
    overwrite : bool
        True if this function should overwrite output_filepath if it already exists
    """
    with tempfile.TemporaryDirectory() as tempdir:
        if input_filepath == output_filepath:
            output_ffmpeg_filepath = join(tempdir, f"{str(uuid.uuid4())}.wav")
        else:
            output_ffmpeg_filepath = output_filepath

        overwrite_flag = "-y" if overwrite else ""

        # If this subprocess run fails then empty blobs (0 bytes) will be created and subsequent steps will fail
        subprocess.run(
            [
                "ffmpeg",
                overwrite_flag,
                "-i",
                f'"{input_filepath}"',
                "-af",
                "dynaudnorm",
                f'"{output_ffmpeg_filepath}"',
            ]
        )

        if input_filepath == output_filepath:
            shutil.move(output_ffmpeg_filepath, output_filepath)
