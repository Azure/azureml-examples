import urllib.request
import json
import os
import ssl
import numpy as np
import mlflow.pyfunc
import pandas as pd
from .demo_utils import get_files_path, get_text
from tqdm import tqdm


class medimageinsight_package:
    """
    The medimageinsight_package class is responsible for deploying MedImageInsight using either an online endpoint or the MLflow architecture.
    By using this class, you can easily deploy MedImageInsight and choose the deployment method that best suits your needs.
    Whether you want to deploy the model as an online endpoint or utilize the MLflow architecture, this class provides a convenient interface to handle the deployment process.
    The core function of this class is to efficiently generate image and text embeddings with MedImageInsight.
    Overall, the medimageinsight_package class provides a template to deploy the MedImageInsight model either as an online endpoint or using the MLflow architecture.
    """

    def __init__(
        self,
        option="run_from_endpoint",
        endpoint_url=None,
        endpoint_key=None,
        model_version=None,
        mlflow_model_path=None,
    ):
        """
        Initializes the MedImageInsight package.

        Parameters:
        - option (str): The option for deployment. Default is 'run_local'.
        - endpoint_url (str): The URL of the endpoint. Required if option is 'run_from_endpoint'.
        - endpoint_key (str): The key to invoke the endpoint. Required if option is 'run_from_endpoint'.
        - model_version (str): The version of the model. Required if option is 'run_from_endpoint'.
        - mlflow_model_path (str): The path to the MLFlow model. Required if option is 'run_local'.
        """
        self.option = option

        if self.option == "run_from_endpoint":
            self.endpoint_url = endpoint_url
            self.endpoint_key = endpoint_key
            self.model_version = model_version
            if not self.endpoint_key:
                raise Exception("A key should be provided to invoke the endpoint")

            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.endpoint_key),
            }
            if self.model_version is not None:
                self.headers["azureml-model-deployment"] = (self.model_version,)
        elif self.option == "run_local":
            self.mlflow_model_path = mlflow_model_path
            if not self.mlflow_model_path:
                raise Exception("A path to the mlflow model should be provided")

            self.mlflow_model = mlflow.pyfunc.load_model(self.mlflow_model_path)

    def allowSelfSignedHttps(allowed):
        """
        Bypasses the server certificate verification on the client side.

        Parameters:
        - allowed (bool): Whether to allow self-signed HTTPS certificates.
        """
        if (
            allowed
            and not os.environ.get("PYTHONHTTPSVERIFY", "")
            and getattr(ssl, "_create_unverified_context", None)
        ):
            ssl._create_default_https_context = ssl._create_unverified_context

    def generate_embeddings(self, data):
        """
        Runs inference to generate embeddings on the model.

        Parameters (Must provide one of the following):
        - data (dict):
            - 'image': The path or data of the image(s).
            - 'text': The text data.

        Returns:
        - embeddings_dict (dict): A dictionary where each key is the name, and the value is another dictionary containing 'image_feature' and/or 'text_feature'.
        """

        embeddings_dict = {}

        # Determine the appropriate function to call based on the option
        if self.option == "run_from_endpoint":
            run_function = self.run_from_endpoint
        elif self.option == "run_local":
            run_function = self.run_from_mlflow
        else:
            raise ValueError(
                f"Invalid option '{self.option}'. Expected 'run_from_endpoint' or 'run_local'."
            )

        # Flags to check if image and/or text data are provided
        has_image = data.get("image") is not None
        has_text = data.get("text") is not None
        params = data["params"] if "params" in data else None

        # Generate embeddings based on provided data
        if has_image and has_text:
            embedding_dict, scale_factor = run_function(
                image=data["image"], text=data["text"], params=params
            )
            for name, feat in embedding_dict.items():
                embeddings_dict.setdefault(name, {})["image_feature"] = feat[
                    "image_feature"
                ]
                embeddings_dict.setdefault(name, {})["text_feature"] = feat[
                    "image_feature"
                ]
        else:
            if has_image:
                image_embedding_dict, scale_factor = run_function(
                    image=data["image"], params=params
                )
                for name, img_feat in image_embedding_dict.items():
                    embeddings_dict.setdefault(name, {})["image_feature"] = img_feat[
                        "image_feature"
                    ]
            if has_text:
                text_embedding_dict, scale_factor = run_function(text=data["text"])
                for name, txt_feat in text_embedding_dict.items():
                    embeddings_dict.setdefault(name, {})["text_feature"] = txt_feat[
                        "text_feature"
                    ]

            return embeddings_dict, scale_factor

    def run_from_mlflow(self, image=None, text=None, params=None):
        """
        Run inference with the MLflow model.

        Parameters:
        - image (str): The path to the image data.
        - text (str): The path to the text data.
        - params (dict): Additional parameters for prediction.
            - image_standardization_jpeg_compression_ratio (int): The JPEG compression ratio for the model input, default: 75.
            - image_standardization_image_size (int): The image size for MedImageInsight model input, default: 512.

        Returns:
        - embeddings_dict (dict): A dictionary where each key is the name,
        and the value is another dictionary containing 'image_feature' and/or 'text_feature'.
        """

        embeddings_dict = {}
        if params is None:
            params = {}

        data_dict = {}

        # Collect image data into a dictionary
        if image is not None:
            # Assuming get_files_path returns a dictionary {name: {'file': image_data, 'index': index}}
            images_data = get_files_path(image)
            for name, data in images_data.items():
                data_dict.setdefault(name, {})["image"] = data["file"]

        # Collect text data into a dictionary
        if text is not None:
            # Assuming get_text returns a dictionary {name: {'text': text_data, 'index': index}}
            texts_data = get_text(text)
            for name, data in texts_data.items():
                data_dict.setdefault(name, {})["text"] = data["text"]

        # Ensure that image and text names match if both are provided
        if image is not None and text is not None:
            assert set(images_data.keys()) == set(
                texts_data.keys()
            ), "Image and text names do not match"
            print("--------Start Generating Image and Text Features--------")
        elif image is not None:
            print("--------Start Generating Image Features--------")
        elif text is not None:
            print("--------Start Generating Text Features--------")
        else:
            raise ValueError("At least one of 'image' or 'text' must be provided.")

        # Process each item in data_dict
        for name, data in tqdm(data_dict.items(), total=len(data_dict)):
            df = pd.DataFrame(
                {"image": [data.get("image", "")], "text": [data.get("text", "")]}
            )
            result = self.mlflow_model.predict(df, params=params)

            embeddings_dict[name] = {}
            if "image_features" in result:
                embeddings_dict[name]["image_feature"] = np.array(
                    result["image_features"][0]
                )
            if "text_features" in result:
                embeddings_dict[name]["text_feature"] = np.array(
                    result["text_features"][0]
                )

        if "scaling_factor" in result:
            scaling_factor = np.array(result["scaling_factor"][0])
        else:
            scaling_factor = None

        if image is not None:
            print("--------Finished All Image Features Generation!!--------")
        if text is not None:
            print("--------Finished All Text Features Prediction!!--------")

        return embeddings_dict, scaling_factor

    def run_from_endpoint(self, image=None, text=None, params=None):
        """
        Deploys the endpoint.

        Parameters:
        - image (str): The path to the image data.
        - text (str): The path to the text data.
        - params (dict): Additional parameters for prediction.
            - image_standardization_jpeg_compression_ratio (int): The JPEG compression ratio for the model input, default: 75.
            - image_standardization_image_size (int): The image size for MedImageInsight model input, default: 512.

        Returns:
        - embeddings_dict (dict): A dictionary where each key is the name,
        and the value is another dictionary containing 'image_feature' and/or 'text_feature'.
        """

        embeddings_dict = {}
        if params is None:
            params = {}

        data_dict = {}

        # Collect image data into a dictionary
        if image is not None:
            images_data = get_files_path(image)
            for name, data in images_data.items():
                data_dict.setdefault(name, {})["image"] = data["file"]

        # Collect text data into a dictionary
        if text is not None:
            texts_data = get_text(text)
            for name, data in texts_data.items():
                data_dict.setdefault(name, {})["text"] = data["text"]

        # Ensure that image and text names match if both are provided
        if image is not None and text is not None:
            assert set(images_data.keys()) == set(
                texts_data.keys()
            ), "Image and text names do not match"
            print("--------Start Generating Image and Text Features--------")
        elif image is not None:
            print("--------Start Generating Image Features--------")
        elif text is not None:
            print("--------Start Generating Text Features--------")
        else:
            raise ValueError("At least one of 'image' or 'text' must be provided.")

        # Process each item in data_dict
        scaling_factor = None
        for name, data in tqdm(data_dict.items(), total=len(data_dict)):
            data_list = [data.get("image", ""), data.get("text", "")]
            request_data = {
                "input_data": {
                    "columns": ["image", "text"],
                    "index": [0],
                    "data": [data_list],
                },
                "params": params,
            }

            body = str.encode(json.dumps(request_data))
            req = urllib.request.Request(self.endpoint_url, body, self.headers)

            try:
                response = urllib.request.urlopen(req)
                result = response.read()

                feature_json = json.loads(result)
                embeddings_dict[name] = {}
                for subj in feature_json:
                    if "image_features" in subj:
                        embeddings_dict[name]["image_feature"] = np.array(
                            subj["image_features"]
                        )
                    if "text_features" in subj:
                        embeddings_dict[name]["text_feature"] = np.array(
                            subj["text_features"]
                        )

                    if "scaling_factor" in subj and scaling_factor is None:
                        scaling_factor = np.array(subj["scaling_factor"])

            except urllib.error.HTTPError as error:
                print(
                    "The embedding generation request failed with status code: "
                    + str(error.code)
                )
                print(error.info())
                print(error.read().decode("utf8", "ignore"))

        if image is not None:
            print("--------Finished All Image Features Generation!!--------")
        if text is not None:
            print("--------Finished All Text Features Generation!!--------")

        return embeddings_dict, scaling_factor
