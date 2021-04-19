"""
Test extract audio from video
"""
import os
import tempfile
from pathlib import Path

import librosa
import pytest
from src.video import extract_audio_from_video

SAMPLE_RATE = 22050


def test_extract_audio_from_video(tmpdir_audio_files):
    """
    Test extract audio from video
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        for audio_filepath in Path(tmpdir_audio_files).glob("*.wav"):
            filename = str(Path(audio_filepath).name)
            test_wav_path = os.path.join(temp_dir, filename)
            extract_audio_from_video(audio_filepath, test_wav_path)
            # Since we relay on ffmpeg to do all the conversion we assume that it is already well tested.
            # So the only thing we should validate is if audio file exists
            assert os.path.exists(test_wav_path)


@pytest.mark.parametrize(
    "sr, start_time, end_time, expected_audio_length",
    [
        (SAMPLE_RATE, "00:00:00", "00:00:00.500", SAMPLE_RATE // 2),
        (SAMPLE_RATE, "00:00:00", "00:00:01", SAMPLE_RATE),
        (SAMPLE_RATE, "00:00:00", "00:00:01.500", 3 * SAMPLE_RATE // 2),
    ],
)
def test_extract_audio_from_video_start_and_end_times(
    tmpdir_audio_files,
    sr: int,
    start_time: str,
    end_time: str,
    expected_audio_length: int,
):
    """
    Test extract audio from video

    Parameters
    ----------
    sr : int
        Parameter to test
    start_time : str
        Parameter to test
    end_time : str
        Paramter to test
    full_video_path : str
        Paramter for validation
    expected_audio_length : int
        Parameter for validation
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        for audio_filepath in Path(tmpdir_audio_files).glob("*.wav"):
            filename = Path(audio_filepath).name
            test_wav_path = os.path.join(temp_dir, filename)
            extract_audio_from_video(
                audio_filepath,
                test_wav_path,
                sr=sr,
                start_time=start_time,
                end_time=end_time,
            )
            audio_clip, sample_rate = librosa.load(test_wav_path)

            assert sample_rate == sr
            assert len(audio_clip) == expected_audio_length
