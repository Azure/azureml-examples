import json
import os
import argparse


def prepare_data_for_online_inference(sample_video_link) -> None:
    """Prepare request json for online inference.

    :param sample_video_links: sample video links
    :type sample_video_links: str
    """
    request_json = {"input_data": {"columns": ["video"], "data": [sample_video_link]}}
    request_file_name = "sample_request_data.json"
    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for video multi object tracking model"
    )
    parser.add_argument(
        "--video_link",
        type=str,
        help="sample demo video link",
        default="https://github.com/open-mmlab/mmtracking/raw/master/demo/demo.mp4",
    )

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    prepare_data_for_online_inference(sample_video_link=args_dict["video_link"])
