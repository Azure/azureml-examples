import os, argparse
import shutil
from azureml.core.run import Run
from pathlib import Path
def main(args):
    print(args.model_input)
    run = Run.get_context()
    model_path=os.path.join(args.model_input,'torch_model.pt')
    dest_dir='./model'
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path,dest_dir)
    run.upload_folder("model",'./model')
    model = run.register_model(model_name="image-classification-torch-model",model_path=dest_dir)
if __name__ == "__main__":
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-input", type=str, help="directory containing trained model"
    )
    args = parser.parse_args()

    # call main function
    main(args)