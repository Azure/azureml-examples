# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Script adapted from: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb
# =========================================================================
import argparse

from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=70)
    parser.add_argument("--per-worker-batch-size", type=int, default=64)
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs",
        help="directory to save the model to",
    )

    args = parser.parse_args()
    from random import random
    accuracy = random()
    
    lines = []
    for variable_name, variable_value in [
        ("epochs", args.epochs),
        ("step-per-epoch", args.steps_per_epoch),
        ("per-worker-batch-size", args.per_worker_batch_size),
        ("accuracy", accuracy)
    ]:
        lines.append(f"{variable_name}: {variable_value}")
        print(lines[-1])

    from azureml.core import Run
    run = Run.get_context()    
    run.log("accuracy", accuracy)

    # Save the model
    (Path(args.model_dir) / "model").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
