# CI setup

The CI is setup with github actions using the on-demand EC2 backend.

This setup currently uses a 4gpu instance p3.8xlarge - to test tp=2, pp=2.

**Unfortunately this only works for PRs created from non-forked branches**


## The workflow file

The workflow file is at `.github/workflows/main.yml`


```
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Start EC2 runner
        id: start-ec2-runner
        uses: machulav/ec2-github-runner@v2
        with:
          mode: start
          github-token: ${{ secrets.GH_PERSONAL_ACCESS_TOKEN }}
          ec2-image-id: ami-0dfaabfa78a779fbc
          ec2-instance-type: p3.8xlarge
          subnet-id: subnet-3502b45e
          security-group-id: sg-e8f46d9d
```

- `ec2-image-id` is the AMI, which has to be created, or copied to the corresponding `aws-region` region the script requests.
- `subnet-id` comes from: https://console.aws.amazon.com/vpc/home?region=us-east-1#subnets:
- `security-group-id` comes from: https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#SecurityGroups:


It was later updated to use a fault-tolerant solution by trying to start the EC2 on 3 different sub-regions to cope with situations where EC2 reports it doesn't have resources to start the desired instance.



## Connect to instance

To pre-install things connect to the instance manually and install what's desired

1. choose and start an EC2 instance
2. connect to it as `ubuntu`, then `sudo su` as the runner runs as `root`. I couldn't find a way around it.
```
ssh -l ubuntu -i "~/.ssh/bigscience-aim.pem" ubuntu@ec2-3-14-127-35.us-east-2.compute.amazonaws.com
```

Once installed, stop the instance.

Then create a new AMI (see below) and update the script using the new AMI.


## Prepare the machine

Steps used to setup fixed software (won't be installed at test time)

- install cuda:
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation

### install fixed packages

- `torch 1.9.0/cu-11.1`

```
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- all kinds of prerequisites
```
pip install transformers
wget https://raw.githubusercontent.com/microsoft/DeepSpeed/master/requirements/requirements.txt -O requirements-ds.txt
pip install -r requirements-ds.txt
wget https://raw.githubusercontent.com/bigscience-workshop/Megatron-DeepSpeed/main/requirements.txt -O requirements-ms.txt
pip install -r requirements-ms.txt

```

- apex - needs a hack to deal with mismatching minor cuda versions (and it takes forever to build), so using this patch:

XXX: this no longer works - had to manually patch pytorch to avoid mismatch failure

```
--- a/setup.py
+++ b/setup.py
@@ -99,6 +99,7 @@ def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
     print(raw_output + "from " + cuda_dir + "/bin\n")

     if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
+        return
         raise RuntimeError("Cuda extensions are being compiled with a version of Cuda that does " +
                            "not match the version used to compile Pytorch binaries.  " +
                            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda) +

```

install it: (it was cloned from `git clone https://github.com/NVIDIA/apex`)

```
cd code/apex
# I copied this script from my setup
./build.sh
```


## make a new AMI image

Once the needed things got installed (and every time anything new is installed) a new AMI must be created (this is like an .iso image snapshot)

1. go to https://us-east-1.console.aws.amazon.com/ec2/v2/home?region=us-east-1#Instances:
2. choose the instance to create a new image from
3. Actions -> Image and Templates -> Create Image

Must ensure it's created in the correct region (same as in script) - or can copy it to the right region.

The process of creating the image can be done while the instance that has been updated is still running.

Just don't forget to turn the instance off when validated it to work.

Finally, once created, the script needs to be updated to that new AMI id (key `ec2-image-id`) in `.github/workflows/main.py`


## Stop instance alarm

It looks like occasionally the instance doesn't stop and continues running.

I added a stop alarm to automatically kill the instance after 1h if util < 10% following the exact instructions from:
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/UsingAlarmActions.html


## Guides

Set up guide: https://github.com/machulav/ec2-github-runner

Launching an EC2 instance:
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html?icmpid=docs_ec2_console

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html

- All available instances: https://aws.amazon.com/ec2/instance-types/
