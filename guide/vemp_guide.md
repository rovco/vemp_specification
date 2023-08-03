# Vaarst Edge Model Package (VEMP) Guide
## Overview
This guide describes in detail how AI models can be built and run within the SubSLAM system. These models must conform to the Vaarst Edge Model Packages (VEMP) specification. The full specification can be found here: [vemp_specification.md](../specification/vemp_specification.md). This section gives an overview of the technologies used by the system.

### Hardware
VEMP models can be executed on three devices within SubSLAM: Topstation (Rackmounted PC or Laptop), Left Camera and Right Camera. It is expected that models will have varying levels of performance depending on the device they are run on, and what resources are available. It is currently only possible to run one model per device. All devices contain hardware acceleration for running AI models.

### ONNX
Open Neural Network Exchange ([ONNX](https://github.com/onnx/onnx)) is an open source format for AI models. It stores both the model architecture and the trained weights, and by design it acts as a common interface between model development and model deployment. [Many common frameworks support ONNX](https://onnx.ai/supported-tools.html), including PyTorch, TensorFlow, Keras, SAS and Matlab.


VEMP specification bases itself on ONNX, but note that SubSLAM does not support all possible variations of ONNX files. VEMP specification defines a subset of the broader ONNX standard, along with a common structure and files.

### TensorRT
[TensorRT](https://developer.nvidia.com/tensorrt) is a high performance SDK for deep learning that optimizes models for NVIDIA hardware. The system uses [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt) runtime to execute models, which only supports a subset of the ONNX operators. Operators refer to a node/layer/module in the model that performs an action. It is important to check the [documentation](https://github.com/onnx/onnx-tensorrt/blob/8.5-GA/docs/operators.md#operator-support-matrix) to ensure the model uses supported operators only.

### Model Architecture
VEMP supports multitask learning, which allows the same model to perform many tasks. These tasks have different expected outputs, see the [specification](../specification/vemp_specification.md) for more details.


## Creating a VEMP
### Steps
High-level steps to create a VEMP:
1.  Export the trained model to ONNX.
2.  [Optional] Edit the ONNX model.
3.  Validate the ONNX model.
4.  Create `model_package.json` and `model_info.json`.

### 1. Export the trained model to ONNX
The process of exporting varies depending on the framework used to train the model.

During the export process, ensure the model conforms to the [VEMP specification](../specification/vemp_specification.md).
Some important things to check:
- Inputs and outputs are the correct types and dimensions.
- The model uses supported operators.
- The model uses a supported Opset version.
- The ONNX file is named `model.onnx`.

#### PyTorch
PyTorch uses tracing to build and export the ONNX graph. There are some caveats, such as always using PyTorch types and not numpy. More info is available at [torch.onnx](https://pytorch.org/docs/stable/onnx.html).

If extra pre-processing or post-processing steps are required, add these to the forward pass of the model before exporting.

#### Other frameworks
Coming soon...

### 2. [Optional] Edit the ONNX model
It is not always possible to export an ONNX model that conforms exactly to the specification. For example, it may be necessary to edit the model to add/remove nodes, rename inputs/outputs, or change a constant initializer.

This is also an opportunity to further simplify and optimize the graph.

#### Graphsurgeon
The [`onnx-graphsurgeon`](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html) Python package
provides a convenient way to create and modify ONNX models. It also provides useful methods to optimise the models:

- `graph.cleanup()` - Removes unused nodes and tensors, i.e. nodes that do not contribute to the model output.
- `graph.toposort()` - Sorts the graph nodes, so each node appears before any nodes that refer to it. This is required for constant folding.
- `graph.fold_constants()` - Optimizes the graph by pre-computing constant expressions. Constant folding may already been performed during the ONNX export process.

#### Onnxsim
The [onnxsim](https://github.com/daquexian/onnx-simplifier) Python package, which uses [onnx-optimizer](https://github.com/onnx/optimizer), performs common optimisations on the ONNX model to remove redundant nodes and simplify the graph. This is a good way to reduce the complexity of the graph. As VEMP specification requires the use of explicit dimensions, onnxsim can infer the shapes for all the edges and write them to the ONNX model. This makes debugging easier as these shapes will be viewable in Netron.

### 3. Validate the ONNX model
#### Netron
[Netron](https://github.com/lutzroeder/netron) is a good tool to visualise, validate and debug the ONNX graph.

This is available on the web browser or an installed AppImage.

#### ONNX Checker
The ONNX Python libraries provide a function [onnx.checker.check_model()](https://onnx.ai/onnx/api/checker.html#check-model) to check the consistency of a model and makes sure it is valid. Using the `full_check=True` parameter, performs a more throughout check, ensuring all shapes can be inferred.

#### Trtexec
Provided as part of TensorRT, this tool builds a TensorRT engine using an ONNX model and executes it with dummy data. This uses same TensorRT runtime as VEMP, and so is a good way to validate a model and it's operators. If the model executes with this tool, it is likely to execute within SubSLAM.

Trtexec is also used to benchmark inference latency. This is an important consideration to prevent the model from impacting the performance of the rest of the SubSLAM system. Performance will vary depending on the device that is executing it. For example, a reduction in performance is expected when running the models on the Cameras compared to the Topstation.

Requirements:
- Nvidia GPU 10XX or greater, with >2GB VRAM.
- Ubuntu 20.04.
- Docker

Run this docker command to execute the model (Note: change the model path):
```
docker run --gpus all -it --rm -v </path/to/model.onnx>:/workspace/model.onnx nvcr.io/nvidia/tensorrt:23.03-py3 /usr/src/tensorrt/bin/trtexec --onnx=/workspace/model.onnx --explicitBatch --workspace=1024 --fp16
```

This command can take up to 10 minutes, depending on the model complexity and hardware available.

It will return `PASSED` when the ONNX model is valid and can be run with TensorRT.

<details>
  <summary>Example of successful trtexec output</summary>

```
[08/03/2023-10:27:37] [I] === Performance summary ===
[08/03/2023-10:27:37] [I] Throughput: 352.359 qps
[08/03/2023-10:27:37] [I] Latency: min = 2.4798 ms, max = 4.60217 ms, mean = 2.83884 ms, median = 2.78979 ms, percentile(90%) = 3.23047 ms, percentile(95%) = 3.39697 ms, percentile(99%) = 3.7373 ms
[08/03/2023-10:27:37] [I] Enqueue Time: min = 2.41815 ms, max = 4.53247 ms, mean = 2.77454 ms, median = 2.7265 ms, percentile(90%) = 3.15723 ms, percentile(95%) = 3.32654 ms, percentile(99%) = 3.66919 ms
[08/03/2023-10:27:37] [I] H2D Latency: min = 0.135254 ms, max = 0.162109 ms, mean = 0.140114 ms, median = 0.13916 ms, percentile(90%) = 0.144043 ms, percentile(95%) = 0.146973 ms, percentile(99%) = 0.155121 ms
[08/03/2023-10:27:37] [I] GPU Compute Time: min = 2.29224 ms, max = 4.41431 ms, mean = 2.64781 ms, median = 2.59973 ms, percentile(90%) = 3.0365 ms, percentile(95%) = 3.20203 ms, percentile(99%) = 3.54492 ms
[08/03/2023-10:27:37] [I] D2H Latency: min = 0.0441895 ms, max = 0.147827 ms, mean = 0.0509221 ms, median = 0.050293 ms, percentile(90%) = 0.0541992 ms, percentile(95%) = 0.0561829 ms, percentile(99%) = 0.0634766 ms
[08/03/2023-10:27:37] [I] Total Host Walltime: 3.00546 s
[08/03/2023-10:27:37] [I] Total GPU Compute Time: 2.80403 s
[08/03/2023-10:27:37] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[08/03/2023-10:27:37] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[08/03/2023-10:27:37] [W] * GPU compute time is unstable, with coefficient of variance = 10.6806%.
[08/03/2023-10:27:37] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[08/03/2023-10:27:37] [I] Explanations of the performance metrics are printed in the verbose logs.
[08/03/2023-10:27:37] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8503] # /usr/src/tensorrt/bin/trtexec --onnx=/workspace/model.onnx --explicitBatch --workspace=1024 --fp16
```

</details>

### 4. Create the remaining files
Refer to the VEMP [specification](../specification/vemp_specification.md) and [schemas](../schemas), to create the `model_package.json` and `model_info.json` files.

These are required for a VEMP to be valid and run within the SubSLAM system. These files should exist in the same directory as the `model.onnx`.

---

## Installing a VEMP
The VEMP must be installed in `/data/` on the target device that will be executing it. For example, a VEMP copied to `/data/my_model` on the Topstation can execute on the Topstation. This directory must contain all three of the required files.

#### Installing on the Topstation
Running third-party models on the Topstation is the recommended approach, as there is more GPU resource available and model packages are easily copied on to the device's filesystem.

#### Installing on the Cameras
Running third-party models on the Cameras is more complicated, as it involves copying the VEMP to the remote filesystem on the Cameras. Please contact [Vaarst support](mailto:support@vaarst.com) for further instructions.

---

## Running a VEMP
Once the VEMP is installed on the target device, it is ready to execute within SubSLAM.

Steps:
1. Enable the Insights feature for the SubSLAM license in use.
2. Open the SubSLAM application.
3. Navigate to the Settings page. Set the filepath to the installed model package, e.g: `/data/my_package/model_package.json`.
4. Start SubSLAM. The model will automatically build and start performing inference. The first execution will take longer, up to 10 minutes. The next execution will be faster as the model will now be in the cache.
5. Models can fail at this step, if they have not been setup correctly or do not conform to the VEMP specification.

See the Diagnostics toolbar at the bottom of SubSLAM to see inference timing metrics.

Please contact [Vaarst support](mailto:support@vaarst.com) for further instructions.

---
<p align=center>
Copyright Vaarst © 2023
<p/>
