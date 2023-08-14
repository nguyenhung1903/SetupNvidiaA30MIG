# Multi-Instance GPU
Multi-Instance GPU (MIG) is an important feature of NVIDIA H100, A100, and A30 Tensor Core GPUs, as it can partition a GPU into multiple instances. Each instance has its own compute cores, high-bandwidth memory, L2 cache, DRAM bandwidth, and media engines such as decoders.

This enables multiple workloads or multiple users to run workloads simultaneously on one GPU to maximize the GPU utilization, with guaranteed quality of service (QoS). A single A30 can be partitioned into up to four MIG instances to run four applications in parallel.

This post walks you through how to use MIG on A30 from partitioning MIG instances to running deep learning applications on MIG instances at the same time.

# A30 MIG profiles
- One GPU instance, with 24 GB of memory
- Two GPU instances, each with 12 GB of memory
- Three GPU instances, one with 12 GB of memory and two with 6 GB
- Four GPU instances, each with 6 GB of memory

# Setup MIG on NVIDIA A30 GPUs
## Step 1: Build MIG Partition Editor (mig-parted)
Clone the repo and build `MIG Partition Editor`: 
```bash
git clone http://github.com/NVIDIA/mig-parted
cd mig-parted
go build ./cmd/nvidia-mig-parted
```

* Notes: Please install **go** first. `sudo snap install go` 

## Step 2: Create a configuration file
```
cat << EOF > a30-example-configs.yaml
version: v1
mig-configs:
  all-disabled:
    - devices: all
      mig-enabled: false

  all-enabled:
    - devices: all
      mig-enabled: true
      mig-devices: {}

  all-1g.6gb:
    - devices: all
      mig-enabled: true
      mig-devices:
        "1g.6gb": 4

  all-2g.12gb:
    - devices: all
      mig-enabled: true
      mig-devices:
        "2g.12gb": 2

  all-balanced:
    - devices: all
      mig-enabled: true
      mig-devices:
        "1g.6gb": 2
        "2g.12gb": 1

  custom-config:
    - devices: [0]
      mig-enabled: true
      mig-devices:
        "1g.6gb": 4
    - devices: [1]
      mig-enabled: true
      mig-devices:
        "2g.12gb": 2
EOF
```

* **all-1g.6gb**: A30 creates 4 MIG instances (Each MIG instance has 6GB)
* **all-2g.12gb**: A30 creates 2 MIG instances (Each MIG instance has 12GB)

## Step 3: Apply configuration file to enable MIG
```bash
~/mig-parted$ sudo ./nvidia-mig-parted apply -f <<filename>> -c all-1g.6gb
```

For example: 

```bash
~/mig-parted$ sudo ./nvidia-mig-parted apply -f a30-example-configs.yaml -c all-1g.6gb
```

## Setup 4: Check status of MIG
```bash
sudo nvidia-smi mig -lgi
```

OUTPUT:
```bash
+-------------------------------------------------------+
| GPU instances:                                        |
| GPU   Name             Profile  Instance   Placement  |
|                          ID       ID       Start:Size |
|=======================================================|
|   0  MIG 1g.6gb          14        3          0:1     |
+-------------------------------------------------------+
|   0  MIG 1g.6gb          14        4          1:1     |
+-------------------------------------------------------+
|   0  MIG 1g.6gb          14        5          2:1     |
+-------------------------------------------------------+
|   0  MIG 1g.6gb          14        6          3:1     |
+-------------------------------------------------------+
``` 

# References

https://github.com/nvidia/mig-parted

https://developer.nvidia.com/blog/dividing-nvidia-a30-gpus-and-conquering-multiple-workloads/
