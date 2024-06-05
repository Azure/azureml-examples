import mltable
import os


def create_ml_table(data_frame, file_name, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    data_path = os.path.join(output_folder, file_name)
    data_frame.to_parquet(data_path, index=False)
    paths = [{"file": data_path}]
    ml_table = mltable.from_parquet_files(paths)
    ml_table.save(output_folder)
