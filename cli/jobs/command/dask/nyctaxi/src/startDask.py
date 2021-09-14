import os
import argparse
import time
from dask.distributed import Client
import sys, uuid
import threading
import subprocess
import socket
import mlflow

from notebook.notebookapp import list_running_servers


def flush(proc, proc_log):
    while True:
        proc_out = proc.stdout.readline()
        if proc_out == "" and proc.poll() is not None:
            proc_log.close()
            break
        elif proc_out:
            sys.stdout.write(proc_out)
            proc_log.write(proc_out)
            proc_log.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jupyter_token", default=uuid.uuid1().hex)
    parser.add_argument("--script")

    args, unparsed = parser.parse_known_args()

    for k, v in os.environ.items():
        if k.startswith("MLFLOW"):
            print(k, v)
    MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")

    print(
        "- env: AZ_BATCHAI_JOB_MASTER_NODE_IP: ",
        os.environ.get("AZ_BATCHAI_JOB_MASTER_NODE_IP"),
    )
    print(
        "- env: AZ_BATCHAI_IS_CURRENT_NODE_MASTER: ",
        os.environ.get("AZ_BATCHAI_IS_CURRENT_NODE_MASTER"),
    )
    print("- env: AZ_BATCHAI_NODE_IP: ", os.environ.get("AZ_BATCHAI_NODE_IP"))
    print("- env: AZ_BATCH_HOST_LIST: ", os.environ.get("AZ_BATCH_HOST_LIST"))
    print("- env: AZ_BATCH_NODE_LIST: ", os.environ.get("AZ_BATCH_NODE_LIST"))
    print("- env: MASTER_ADDR: ", os.environ.get("MASTER_ADDR"))
    print("- env: MASTER_PORT: ", os.environ.get("MASTER_PORT"))
    print("- env: RANK: ", os.environ.get("RANK"))
    print("- env: LOCAL_RANK: ", os.environ.get("LOCAL_RANK"))
    print("- env: NODE_RANK: ", os.environ.get("NODE_RANK"))
    print("- env: WORLD_SIZE: ", os.environ.get("WORLD_SIZE"))

    rank = os.environ.get("RANK")
    ip = socket.gethostbyname(socket.gethostname())
    master = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")

    print("- my rank is ", rank)
    print("- my ip is ", ip)
    print("- master is ", master)
    print("- master port is ", master_port)

    scheduler = master + ":8786"
    dashboard = master + ":8787"
    print("- scheduler is ", scheduler)
    print("- dashboard is ", dashboard)

    print("args: ", args)
    print("unparsed: ", unparsed)
    print("- my rank is ", rank)
    print("- my ip is ", ip)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    print("free disk space on /tmp")
    os.system(f"df -P /tmp")

    if str(rank) == "0":
        mlflow.log_param("headnode", ip)
        mlflow.log_param(
            "cluster",
            "scheduler: {scheduler}, dashboard: {dashboard}".format(
                scheduler=scheduler, dashboard=dashboard
            ),
        )

        cmd = (
            "jupyter lab --ip 0.0.0.0 --port 8888"
            + " --NotebookApp.token={token}"
            + " --allow-root --no-browser"
        ).format(token=args.jupyter_token)
        os.environ["MLFLOW_RUN_ID"] = MLFLOW_RUN_ID
        jupyter_log = open("logs/jupyter_log.txt", "w")
        jupyter_proc = subprocess.Popen(
            cmd.split(),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        jupyter_flush = threading.Thread(target=flush, args=(jupyter_proc, jupyter_log))
        jupyter_flush.start()

        # while not list(list_running_servers()):
        #    time.sleep(5)

        # jupyter_servers = list(list_running_servers())
        # assert (len(jupyter_servers) == 1), "more than one jupyter server is running"

        mlflow.log_param(
            "jupyter", "ip: {ip_addr}, port: {port}".format(ip_addr=ip, port="8888")
        )
        mlflow.log_param("jupyter-token", args.jupyter_token)

        cmd = (
            "dask-scheduler "
            + "--port "
            + scheduler.split(":")[1]
            + " --dashboard-address "
            + dashboard
        )
        print(cmd)
        os.environ["MLFLOW_RUN_ID"] = MLFLOW_RUN_ID
        scheduler_log = open("logs/scheduler_log.txt", "w")
        scheduler_proc = subprocess.Popen(
            cmd.split(),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        scheduler_flush = threading.Thread(
            target=flush, args=(scheduler_proc, scheduler_log)
        )
        scheduler_flush.start()

        cmd = "dask-worker " + scheduler
        print(cmd)
        os.environ["MLFLOW_RUN_ID"] = MLFLOW_RUN_ID
        worker_log = open("logs/worker_{rank}_log.txt".format(rank=rank), "w")
        worker_proc = subprocess.Popen(
            cmd.split(),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        worker_flush = threading.Thread(target=flush, args=(worker_proc, worker_log))
        worker_flush.start()

        if args.script:
            command_line = " ".join(["python", args.script] + unparsed)
            print("Launching:", command_line)

            os.environ["MLFLOW_RUN_ID"] = MLFLOW_RUN_ID
            driver_log = open("logs/driver_log.txt", "w")
            driver_proc = subprocess.Popen(
                command_line.split(),
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            driver_flush = threading.Thread(
                target=flush, args=(driver_proc, driver_log)
            )
            driver_flush.start()

            # Wait until process terminates (without using p.wait())
            # while driver_proc.poll() is None:
            #    # Process hasn't exited yet, let's wait some
            #    time.sleep(0.5)

            print("waiting for driver process to terminate")
            driver_proc.wait()

            exit_code = driver_proc.returncode
            print("process ended with code", exit_code)
            print("killing scheduler, worker and jupyter")

            jupyter_proc.kill()
            scheduler_proc.kill()
            worker_proc.kill()
            exit(exit_code)
        else:
            flush(scheduler_proc, scheduler_log)
    else:
        cmd = "dask-worker " + scheduler
        print(cmd)
        os.environ["MLFLOW_RUN_ID"] = MLFLOW_RUN_ID
        worker_log = open("logs/worker_{rank}_log.txt".format(rank=rank), "w")
        worker_proc = subprocess.Popen(
            cmd.split(),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        flush(worker_proc, worker_log)
