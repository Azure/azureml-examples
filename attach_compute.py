

import sys, time
from azureml.core.compute import KubernetesCompute, ComputeTarget
from azureml.core.workspace import Workspace
from azureml.exceptions import ComputeTargetException

INSTANCE_TYPES = {
    "STANDARD_DS3_V2": {
        "nodeSelector": None,
        "resources": {
            "requests": {
                "cpu": "2",
                "memory": "4Gi",
            }
        }
    },
    "STANDARD_NC12": {
        "nodeSelector": None,
        "resources": {
            "requests": {
                "cpu": "8",
                "memory": "64Gi",
                "nvidia.com/gpu": 2
            }
        }
    },
    "STANDARD_NC6": {
        "nodeSelector": None,
        "resources": {
            "requests": {
                "cpu": "3",
                "memory": "32Gi",
                "nvidia.com/gpu": 1
            }
        }
    }
}

def main():

    print("args:", sys.argv)

    sub_id=sys.argv[1]
    rg=sys.argv[2]
    ws_name=sys.argv[3]
    k8s_compute_name = sys.argv[4]
    resource_id = sys.argv[5]
    instance_type = sys.argv[6]

    ws = Workspace.get(name=ws_name,subscription_id=sub_id,resource_group=rg)

    for i in range(10):
        try:
            try:
                # check if already attached
                k8s_compute = KubernetesCompute(ws, k8s_compute_name)
                print("compute already existed. will detach and re-attach it")
                k8s_compute.detach()
            except ComputeTargetException:
                print("compute not found")

            k8s_attach_configuration = KubernetesCompute.attach_configuration(resource_id=resource_id, default_instance_type=instance_type, instance_types=INSTANCE_TYPES)
            k8s_compute = ComputeTarget.attach(ws, k8s_compute_name, k8s_attach_configuration)
            k8s_compute.wait_for_completion(show_output=True)
            print("compute status:", k8s_compute.get_status())

            return 0
        except Exception as e:
            print("ERROR:", e)
            print("Will sleep 30s. Epoch:", i)
            time.sleep(30)

    sys.exit(1)

if __name__ == "__main__":
    main()



