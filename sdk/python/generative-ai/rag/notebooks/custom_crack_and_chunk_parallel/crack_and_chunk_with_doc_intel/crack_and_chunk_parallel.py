import os
import pandas as pd
import numpy as np
import argparse
import mlflow
import traceback
from pathlib import Path
import shutil

from azureml.rag.documents.cracking import file_extension_loaders
from azureml.rag.tasks.crack_and_chunk import (
    crack_and_chunk_arg_parser,
    str2bool,
)
from azureml.rag.tasks.crack_and_chunk import main as main_crack_and_chunk
from azureml.rag.utils.connections import get_connection_by_id_v2
from azureml.rag.utils.logging import (
    get_logger,
    safe_mlflow_start_run,
    track_activity,
)

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from document_intelligence_loader import DocumentIntelligencePDFLoader

logger = get_logger("crack_and_chunk_document_intelligence")


def init():
    global parser
    global args
    parser = crack_and_chunk_arg_parser()

    # Need to add use_layout, otherwise will get error saying no attribute of use_layout.
    parser.add_argument(
        "--use_layout", type=str2bool, default=False, help="Use layout for PDF cracking"
    )
    args, _ = parser.parse_known_args()

    global INPUT_DATA
    global INPUT_GLOB
    global ALLOWED_EXTENSIONS
    global OUTPUT_CHUNKS
    global CHUNK_SIZE
    global CHUNK_OVERLAP
    global USE_RCTS
    global DATA_SOURCE_URL
    global DOCUMENT_PATH_REPLACEMENT_REGEX
    global DOC_INTEL_CONNECTION_ID
    global USE_LAYOUT

    INPUT_DATA = args.input_data
    INPUT_GLOB = args.input_glob
    ALLOWED_EXTENSIONS = args.allowed_extensions
    OUTPUT_CHUNKS = args.output_chunks
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    USE_RCTS = args.use_rcts
    DATA_SOURCE_URL = args.data_source_url
    DOCUMENT_PATH_REPLACEMENT_REGEX = args.document_path_replacement_regex
    DOC_INTEL_CONNECTION_ID = args.doc_intel_connection_id
    USE_LAYOUT = args.use_layout


def main_function(args, logger, activity_logger, mini_batch):
    args.input_glob = str(INPUT_GLOB)[1:-1]
    args.allowed_extensions = ALLOWED_EXTENSIONS
    args.output_chunks = OUTPUT_CHUNKS
    args.chunk_size = int(CHUNK_SIZE)
    args.chunk_overlap = int(CHUNK_OVERLAP)
    args.use_rcts = str2bool(USE_RCTS)
    args.data_source_url = DATA_SOURCE_URL
    args.document_path_replacement_regex = DOCUMENT_PATH_REPLACEMENT_REGEX
    args.doc_intel_connection_id = str(DOC_INTEL_CONNECTION_ID)[1:-1]
    args.use_layout = USE_LAYOUT

    # The following statement is needed, otherwise args can't go through the subfolder of input blob container.
    # copy is also needed, otherwise the input data is empty.
    my_path = Path.cwd()

    for files in mini_batch:
        shutil.copy(files, my_path)

    args.input_data = my_path
    # Needs to set use_layout to True here, otherwise default is False
    args.use_layout = True

    # Here is chance to update the following parameters if you want, otherwise can comment out
    """
    args.max_sample_files=-1
    args.output_format="csv"
    args.custom_loader=None
    args.output_title_chunk=None
    args.openai_api_version=None
    args.openai_api_type=None
    """

    if args.doc_intel_connection_id:
        document_intelligence_connection = get_connection_by_id_v2(
            args.doc_intel_connection_id
        )
        print(
            "here is the document intelligence connection: ",
            document_intelligence_connection,
        )
        os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"] = document_intelligence_connection[
            "properties"
        ]["metadata"]["endpoint"]
        os.environ["DOCUMENT_INTELLIGENCE_KEY"] = document_intelligence_connection[
            "properties"
        ]["credentials"]["keys"]["api_key"]
        os.environ["AZURE_AI_DOCUMENT_INTELLIGENCE_USE_LAYOUT"] = str(args.use_layout)

        DocumentIntelligencePDFLoader.document_intelligence_client = (
            DocumentAnalysisClient(
                endpoint=document_intelligence_connection["properties"]["metadata"][
                    "endpoint"
                ],
                credential=AzureKeyCredential(
                    document_intelligence_connection["properties"]["credentials"][
                        "keys"
                    ]["api_key"]
                ),
                headers={"x-ms-useragent": "crack_and_chunk_parallel/1.0.0"},
            )
        )
        DocumentIntelligencePDFLoader.use_layout = args.use_layout
    else:
        print("Hello, doc_intel_connection_id not valid or not existing")
        raise ValueError("doc_intel_connection_id is required")

    # Override default `.pdf` loader to use Azure AI Document Intelligence
    file_extension_loaders[".pdf"] = DocumentIntelligencePDFLoader
    main_crack_and_chunk(args, logger, activity_logger)


def run(mini_batch):
    with track_activity(
        logger, "crack_and_chunk_document_intelligence"
    ) as activity_logger, safe_mlflow_start_run(logger=logger):
        try:
            main_function(args, logger, activity_logger, mini_batch)
        except Exception:
            activity_logger.error(
                f"crack_and_chunk_document_intelligence failed with exception: {traceback.format_exc()}"
            )
            raise

    return mini_batch
