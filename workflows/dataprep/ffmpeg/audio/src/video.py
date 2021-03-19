"""
Video preprocessing module
"""
import os
import subprocess
from typing import Optional

from .aml import get_logger
from .common import reuse
from .timestamp import format_timestamp_to_seconds

log = get_logger(__name__)


@reuse
def extract_audio_from_video(
    input_video_path: str,
    output_audio_path: str,
    sr: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    use_gpu: bool = False,
    overwrite: bool = False,
):
    """Extracts audio track from video file and saves it as wav file.
    It will also change sampling rate based on `sr` param provided.
    For video->audio extraction ffmpeg library is used.

    If a start_time and end_time parameter are passed, the extracted audio will only
    be in the segment between those times

    Parameters
    ----------
    input_video_path : str
        Video file path.
    output_audio_path : str
        Output wav file path.
    sr : int, optional
        Target sampling rate.
        If `None`, then the default sampling rate of the input video will be used
    start_time : str, optional
        Start time to extract audio from video in HH:MM:SS.sss format
        If `None` then the whole audio will be extracted
    end_time : str, optional
        End time to extract audio from video in HH:MM:SS.sss format.
        If `None` then the whole audio will be extracted
    use_gpu : bool, optional
        Use ffmpeg enabled with GPU.
        By default False.
    overwrite : bool, optional
        Overwrite audio files at output_audio_path.
        By default False.
    """
    input_video_path = str(input_video_path)
    output_audio_path = str(output_audio_path)

    # Create dir if not exists
    dir_name = os.path.dirname(output_audio_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    target_sr_flag = f"-ar {sr}" if sr is not None else ""

    overwrite_flag = "-y" if overwrite else ""
    cuda_flag = "-hwaccel cuda" if use_gpu else ""

    if start_time is not None and end_time is not None:
        end_time = format_timestamp_to_seconds(end_time) - format_timestamp_to_seconds(
            start_time
        )

        # Timestamp flag will preceed the input -i see https://trac.ffmpeg.org/wiki/Seeking for details
        # pylint: disable=line-too-long
        command = [
            "ffmpeg",
            cuda_flag,
            "-ss",
            start_time,
            "-i",
            input_video_path,
            "-t",
            end_time,
            "-vn",
            "-acodec",
            "pcm_s161e",
            str(target_sr_flag),
            "-ac",
            "1",
            overwrite_flag,
            output_audio_path,
        ]
    else:
        # prepare command for ffmpeg
        command = [
            "ffmpeg",
            cuda_flag,
            "-i",
            input_video_path,
            str(target_sr_flag),
            overwrite_flag,
            output_audio_path,
        ]

    log.info("Executing ffmpeg command: %s", command)
    subprocess.run(command)
