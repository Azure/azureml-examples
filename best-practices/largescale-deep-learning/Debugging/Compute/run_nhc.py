import subprocess
import requests
import os

check_descriptions = {
    ####################################################################
    ###
    ### Hardware checks
    ###
    "check_hw_cpuinfo": 'Compares the properties of the OS-detected CPU(s) \
to the expected values for the SKU type to ensure that the correct \
number of physical sockets, execution cores, and \
"threads" (or "virtual cores") are present and functioning on the system',
    "check_hw_physmem": "Compares the amount of physical memory (RAM) \
present in the system with the minimum and maximum expceted values for the SKU type.",
    "check_hw_swap": "Compares the total system virtual memory (swap) \
size with the minimum and maximum expceted values for the SKU type.",
    "check_hw_ib": "Determines whether or not an active Infiniband link \
is present with the expected data rate. Also checks that the Infiniband \
device type is correct and that the kernel drivers and userspace libraries \
are the same OFED version",
    "check_hw_eth": "Verifies that a particular Ethernet device is available.",
    "check_hw_topology": "Checks that the hardware topology matches the \
expected topology for the SKU type.",
    #######################################################################
    #####
    ##### GPU checks
    #####
    "check_gpu_count": "Checks that the GPU count detected by nvidia-smi is \
equal to the expected GPU count of this SKU type.",
    "check_gpu_xid": "Checks for GPU xid errors in the kernel log. These \
errors can occur if the driver is programming the GPU incorrectly or there \
is a corruption of the commands sent to the GPU.",
    "check_nvsmi_healthmon": "Runs the nvidia healthmon test. nvidia-healthmon \
focuses on checking for software and system configuration issues with GPUs.",
    "check_gpu_bw": "This check tests GPU bandwidth using NVBandwidth and \
compares the results to the expected bandwidth for the SKU type.",
    "check_gpu_ecc": "Checks for GPU Error Correcting Code(ECC) \
errors. These indicate issues with GPU memory.",
    "check_gpu_clock_throttling": "Checks the GPU clock throttling reasons \
for unexpected behavior.",
    "check_nccl_allreduce": "Runs the NCCL allreduce test to test that the \
speed of inter-GPU communication is equal to the expected speed. \
This test failing can indicate issues with NVLink.",
    "check_nvlink_status": "Checks that NVLink is enabled.",
    #######################################################################
    ###
    ### Performance check
    ###
    "check_cpu_stream": "Check that the CPU memory bandwidth matches the \
expected value for the SKU type.",
    ########################################################################
    #####
    ##### Additional IB checks
    #####
    "check_ib_bw_non_gdr": "Checks that the Infiniband bandwidth matches \
the expected value for the SKU type.",
    "check_ib_bw_gdr": "Checks that the Infiniband bandwidth matches \
the expected value for the SKU type.",
    "check_nccl_allreduce_ib_loopback": "Checks for Infiniband issues by \
running NCCL allreduce and disabling NCCL shared memory.",
    "check_ib_link_flapping": "Checks for Infiniband link flapping within \
a specified time interval. None should occur.",
}


# read output log and raise errors
def parse_output(output_file):
    with open(output_file) as output_log:
        read_output_log = output_log.read()
        index = 0
        full_errors = ""
        while index != -1:
            index = read_output_log.find("ERROR:", index + 1)
            if index == -1:
                break
            first_endline = read_output_log.find("\n", index + 5)
            error_message = read_output_log[index : first_endline + 1]
            if error_message not in full_errors:
                full_errors = full_errors + error_message
        if full_errors and os.getenv("KICK_BAD_NODE", "False").lower() in (
            "true",
            "1",
            "t",
        ):
            # Overload disk to kick bad node off cluster
            print(
                "Errors detected during node health checks. Kicking bad node off cluster."
            )
            device = "/dev/root"
            mount_point = "/mnt/my_mount_point"
            if not os.path.exists(mount_point):
                os.makedirs(mount_point)
            os.system(f"mount {device} {mount_point}")
            os.system(f"ls /mnt/")
            command = "fallocate -l 3T /mnt/my_mount_point/bigfile"
            process = subprocess.Popen(
                command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output, error = process.communicate()
            print(output)
            print(error)
        if full_errors:
            raise Exception(
                "Failures were found while running the node health checks. Please see the std_log_process.txt files under the 'outputs and logs' tab of the job for more information."
                + full_errors
            )


# open conf file to read in tests being performed for this SKU type
def parse_conf(conf_file_name):
    with open(conf_file_name) as conf_file:
        read_conf_file = conf_file.read()
        index = 0
        checks = []
        while index != -1:
            index = read_conf_file.find("* ||", index + 1)
            if index == -1:
                break
            first_endline = read_conf_file.find("\n", index + 5)
            first_space = read_conf_file.find(" ", index + 5)
            if first_endline != -1 and first_space != -1:
                check_end = (
                    first_endline if first_endline < first_space else first_space
                )
                check = read_conf_file[index + 5 : check_end]
            elif first_space == -1:
                check = read_conf_file[index + 5 :]
            else:
                check = read_conf_file[index + 5 : first_space]
            if check not in checks:
                checks.append(check)
        return checks


# Print explanations for all checks being run
def print_check_desc(checks):
    print("| CHECK NAME" + (" " * 30) + "| CHECK DESCRIPTION")
    print("|" + ("-" * 41) + "|" + ("-" * 60))
    for check in checks:
        max_check_name_length = 40
        desc_cutoff = outputed_desc = 60
        if check in check_descriptions:
            print(
                "| "
                + check
                + (" " * (max_check_name_length - len(check)))
                + "| "
                + check_descriptions[check]
            )


if __name__ == "__main__":
    # change to nhc directory
    os.chdir("/root/")
    # Run nhc script
    subprocess.run(["sudo", "-E", "bash", "/azure-nhc/aznhc-entrypoint.sh"])
    # get sku type
    r = requests.get(
        "http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-01-01&format=text",
        headers={"Metadata": "true"},
        timeout=10,
    )
    sku = r.content.decode("utf-8")[9:].lower()
    checks = parse_conf(f"/azure-nhc/default/conf/{sku}.conf")
    print_check_desc(checks)
    parse_output("/azure-nhc/output/aznhc.log")
