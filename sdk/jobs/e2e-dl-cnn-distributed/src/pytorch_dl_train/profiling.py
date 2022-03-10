import os
import time
import logging

def markdown_trace_handler(dir_name: str, rank: int = 0):
    """This handler can be used inside torch.profiler call to output
    tables in markdown format"""
    def handler_fn(prof) -> None:
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)
        
        # Note: trying to identify a unique name for the file
        file_name =  os.path.join(dir_name, "step{}_rank{}_t{}.md".format(prof.step_num, rank, int(time.time() * 1000)))

        logging.getLogger(__name__).info(f"Exporting profiler trace as markdown at {file_name}")
        # generate report in markdown format
        markdown = [ "# Pytorch Profiler report" ]

        markdown.append("## Average by cuda time")
        markdown.append("```")
        markdown.append(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        markdown.append("```")

        with open(file_name, "w") as out_file:
            out_file.write("\n".join(markdown))

    return handler_fn
