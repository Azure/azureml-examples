import os
import traceback

from azureml.rag.documents.cracking import file_extension_loaders
from azureml.rag.tasks.crack_and_chunk import (
    __main__,
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


def main(args, logger, activity_logger):
    if args.doc_intel_connection_id:
        document_intelligence_connection = get_connection_by_id_v2(
            args.doc_intel_connection_id
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
            )
        )
        DocumentIntelligencePDFLoader.use_layout = args.use_layout
    else:
        raise ValueError("doc_intel_connection_id is required")

    # Override default `.pdf` loader to use Azure AI Document Intelligence
    file_extension_loaders[".pdf"] = DocumentIntelligencePDFLoader

    main_crack_and_chunk(args, logger, activity_logger)


def main_wrapper(args, logger):
    with track_activity(
        logger, "crack_and_chunk_document_intelligence"
    ) as activity_logger, safe_mlflow_start_run(logger=logger):
        try:
            main(args, logger, activity_logger)
        except Exception:
            activity_logger.error(
                f"crack_and_chunk_document_intelligence failed with exception: {traceback.format_exc()}"
            )
            raise


if __name__ == "__main__":
    parser = crack_and_chunk_arg_parser()

    parser.add_argument(
        "--doc_intel_connection_id",
        type=str,
        help="Custom Connection to use for Document Intelligence",
    )
    parser.add_argument(
        "--use_layout", type=str2bool, default=False, help="Use layout for PDF cracking"
    )

    __main__(parser, main_wrapper)
