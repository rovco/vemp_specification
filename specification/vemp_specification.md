# Vaarst Edge Model Package (VEMP) Specification
A VEMP is a directory that contains three files: `model_package.json`, `model_info.json` and `model.onnx`. These files must conform to the following specification.

---

## Model Package
The `model_package.json` file describes how the model will be executed, such as specifying which image stream is used for input or the cache location. This is designed to be easily configurable before runtime.

This file must conform to the schema: [`model_package.schema.json`](../schemas/model_package.schema.json)

<details>
  <summary>Fields</summary>

| Key | Description | Required |
| --- | --- | --- |
| version | The VEMP specification version that the model package conforms to. This version needs to be compatible with the version of SubSLAM. | Yes |
| input.camera | Use 'left', 'right' or 'any' images from the cameras, as inputs into the model. | Yes |
| input.resize | The mode used to resize the input camera images to match the model input dimensions:<br />Squish - Resizes the input, this may distort the image but no input pixels are discarded.<br />Crop - Crops the input, some pixels may be discarded. | Yes |
| output.raw | Emit outputs as structs, which can be used by internal libraries. | Yes |
| output.ros | Emit outputs as [ROS](https://wiki.ros.org/) messages. | Yes |
| download.source | Location to download models from. For internal use only. | No |
| download.id | Identifier of model to download from source. For internal use only. | No |
| cache_dir | File path to directory where the cache is saved. | No |

Refer to the [`model_package.schema.json`](../schemas/model_package.schema.json) for more information on these fields, such as types and optionality.

</details>

<details>
  <summary>Example</summary>

```json
{
    "version": "1.2.0",
    "input": {
      "camera": "any",
      "resize": "squish"
    },
    "output": {
      "raw": true,
      "ros": true
    },
  }
```
</details>

---

## Model Info
The `model_info.json` file contains metadata specific to the model, such as the task details.

This file must conform to schema: [`model_info.schema.json`](../schemas/model_info.schema.json)

<details>
  <summary>Fields</summary>

| Key | Description | Required |
| --- | --- | --- |
| type | Type of the model being used, currently only supports `multi_task`. | Yes |
| tasks | List of information for each task in the model, such as class names. The keys of these tasks must correspond the names outputs in the model. The number of classes should also match the number of possible outputs from that tasks. | Yes |

Refer to the [`model_info.schema.json`](../schemas/model_info.schema.json) for more information on these fields, such as types and optionality.

</details>

<details>
  <summary>Example</summary>

This example uses all three tasks: segmentation, detection and classification.
```json
{
    "type": "multi_task",
    "tasks": {
        "underwater_segmentation_task": {
            "type": "segmentation",
            "classes": [
                "nothing",
                "water",
                "asset",
                "ground",
            ]
        },
        "windfarm_parts_detector_task": {
            "type": "detection",
            "classes": [
                "anode",
                "bolt",
                "weld",
            ]
        },
        "diver_classifier_task": {
            "type": "classification",
            "classes": [
                true,
                false
            ]
        },
    }
}
```
</details>

---

## ONNX Model
The `model.onnx` file is the trained model, containing the graph and weights. The ONNX specification is very broad and customisable, so the model must conform to the specific requirements below to ensure compatibility with our system.

##### Requirements
- Uses ONNX Opset version <= 13.
- Uses onnx-tensorrt v8.0.1 [supported operators](https://github.com/onnx/onnx-tensorrt/blob/8.0-GA/docs/operators.md#operator-support-matrix).
- Uses fixed dimensions and a batch size of 1.
- Input and Outputs buffers conform to the specification below.

#### Input
Input images from cameras are pushed to this buffer. A model must have one Input Buffer. If normalization is required, this must be done as part of the ONNX.

| | |
| --- | --- |
| Name | `input` |
| Shape | `[1(batch size), 3(rgb), height, width]` |
| Type | `float32` |
| Range | `[0, 1]` |

#### Outputs
A model must have 1 or more tasks. For any given task, the model must contain all the required output buffers.
<details>
  <summary>Classification Task</summary>
<br/>
Image classification output, predicting a class and confidence for a given image.

###### 1. Class Index Buffer
| | |
| --- | --- |
| Name | `<task_name>_class_index` |
| Shape | `[1(batch size)]` |
| Type | `int32` |
| Range | `[0, class count]` |

###### 2. Confidence Buffer
| | |
| --- | --- |
| Name | `<task_name>_confidence` |
| Dims | `[1(batch size)]` |
| Type | `float32` |
| Range | `[0.0, 1.0]` |

</details>

<details>
  <summary>Segmentation Task</summary>
<br/>
Semantic segmentation output, predicting a segmentation map for a given image.

###### 1. Segmentation Map Buffer
| | |
| --- | --- |
| Name | `<task_name>_segmentation_map` |
| Shape | `[1(batch size), height, width]` |
| Type | `int32` |
| Range | `[0, class count]` |

*Hint: A map of indices from ArgMaxed predictions.*

</details>

<details>
<br/>
  <summary>Detection Task</summary>

Object detection output, predicting bounding boxes for a given image. NMS should already be applied to the output.

###### 1. Boxes Buffer
| | |
| --- | --- |
| Name | `<task_name>_boxes` |
| Shape | `[1(batch size), boxes count, 4(x1, y1, x2, y2)]` |
| Type | `float32` |
| Range | `[0, width/height]` un-normalised |

*Hint: The positions of the detected bounding boxes.*

###### 2. Classification Buffer
| | |
| --- | --- |
| Name | `<task_name>_classification` |
| Shape | `[batch_size=1, classes_count, max_boxes]` |
| Type | `float32` |
| Range | `[0, 1]` |

*Hint: Classifications for each detected box.*

###### 3. NMS Indices Buffer

| | |
| --- | --- |
| Name | `<task_name>_nms_indices` |
| Shape | `[max output boxes per class, 3(batch index, class index, box index)]` |
| Type | `int32` |
| Range | `[0, max boxes]` |

*Hint: The indices produced by NonMaximumSupression for the bounding box predictions. More info see [ONNX operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression) and [TensorRT ONNX operators](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)*
</details>

---
<p align=center>
Copyright Vaarst © 2022
<p/>