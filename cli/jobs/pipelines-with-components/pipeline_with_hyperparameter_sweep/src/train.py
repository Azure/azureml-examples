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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--first-layer-neurons", type=int, default=40)
    parser.add_argument("--second-layer-neurons", type=int, default=20)
    parser.add_argument("--third-layer-neurons", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--momentum", type=float, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--f1", type=float, default=0.5)
    parser.add_argument("--f2", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--model-output",
        type=str,
        default="outputs",
        help="directory to save the model to",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="inputs",
        help="data for training",
    )
    
    args = parser.parse_args()
    from random import random
    accuracy = random()
    lines = []
    for param_name, param_value in [
        ("training_data", args.training_data),
        ("batch_size", args.batch_size),
        ("first_layer_neurons", args.first_layer_neurons),
        ("second_layer_neurons", args.second_layer_neurons),
        ("third_layer_neurons", args.third_layer_neurons),
        ("epochs", args.epochs),
        ("momentum", args.momentum),
        ("weight_decay", args.weight_decay),
        ("learning_rate", args.learning_rate),
        ("f1", args.f1),
        ("f2", args.f2),
        ("model_output", args.model_output),
        ("random_seed", args.random_seed), 
        ("accuracy", accuracy), 
    ]:
        lines.append(f"{param_name}: {param_value}")
        print(lines[-1])

    from azureml.core import Run
    run = Run.get_context()
    run.log("accuracy", accuracy)

    # Do the train and save the trained model as a file into the output folder.
    # Here only output a dummy data for demo.
    (Path(args.model_output) / "model").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
