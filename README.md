# Depth Estimation for AD 
> We reproduce the results of Depth-Anything via HP Workstation

## Performance

Here we reproduce and compare our Depth Anything with the previously best MiDaS v3.1 BEiT<sub>L-512</sub> model.

Please note that the latest MiDaS is also trained on KITTI and NYUv2, while we do not.

| Method | Params | KITTI || NYUv2 || Sintel || DDAD || ETH3D || DIODE ||
|-|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| | | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ |
| MiDaS | 345.0M | 0.127 | 0.850 | 0.048 | *0.980* | 0.587 | 0.699 | 0.251 | 0.766 | 0.139 | 0.867 | 0.075 | 0.942 | 
| **Ours-S** | 24.8M | 0.080 | 0.936 | 0.052 | 0.972 | 0.462 | 0.739 | 0.245 | 0.767 | 0.127 | **0.885** | 0.076 | 0.939 |
| **Ours-B** | 97.5M | *0.080* | *0.939* | *0.046* | 0.978 | **0.432** | *0.756* | *0.232* | *0.786* | **0.126** | *0.884* | *0.069* | *0.946* |
| **Ours-L** | 335.3M | **0.076** | **0.947** | **0.043** | **0.981** | *0.458* | **0.760** | **0.230** | **0.789** | *0.127* | 0.882 | **0.066** | **0.952** |

We highlight the **best** and *second best* results in **bold** and *italic* respectively (**better results**: AbsRel $\downarrow$ , $\delta_1 \uparrow$).

## Pre-trained models

We utilize three models of varying scales for robust relative depth estimation:

| Model | Params | Inference Time on V100 (ms) | A100 | RTX4090 ([TensorRT](https://github.com/spacewalk01/depth-anything-tensorrt)) |
|:-|-:|:-:|:-:|:-:|
| Depth-Anything-Small | 24.8M | 12 | 8 | 3 |
| Depth-Anything-Base | 97.5M | 13 | 9 | 6 |
| Depth-Anything-Large | 335.3M | 20 | 13 | 12 |

Note that the V100 and A100 inference time (*without TensorRT*) is computed by excluding the pre-processing and post-processing stages, whereas the last column RTX4090 (*with TensorRT*) is computed by including these two stages (please refer to [Depth-Anything-TensorRT](https://github.com/spacewalk01/depth-anything-tensorrt)).

The pre-trained models can be easily loaded by:
```python
from depth_anything.dpt import DepthAnything

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
```

Depth Anything is also supported in [``transformers``](https://github.com/huggingface/transformers). You can use it for depth prediction within [3 lines of code](https://huggingface.co/docs/transformers/main/model_doc/depth_anything) (credit to [@niels](https://huggingface.co/nielsr)).


## Usage 

### Installation

```bash
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -r requirements.txt
```

### Running

```bash
python run.py --encoder <vits | vitb | vitl> --img-path <img-directory | single-img | txt-file> --outdir <outdir> [--pred-only] [--grayscale]
```
Arguments:
- ``--img-path``: you can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- ``--pred-only`` is set to save the predicted depth map only. Without it, by default, we visualize both image and its depth map side by side.
- ``--grayscale`` is set to save the grayscale depth map. Without it, by default, we apply a color palette to the depth map.

For example:
```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

**If you want to use Depth Anything on videos:**
```bash
python run_video.py --encoder vitl --video-path assets/examples_video --outdir video_depth_vis
```

### Gradio demo

To use our gradio demo locally:

```bash
python app.py
```

You can also try our [online demo]([https://huggingface.co/spaces/LiheYoung/Depth-Anything](https://huggingface.co/spaces/wbjsamuel/Depth-Estimation-AD/tree/main)).

### Import Depth Anything to your project

If you want to use Depth Anything in your own project, you can simply follow [``run.py``](run.py) to load our models and define data pre-processing. 

<details>
<summary>Code snippet (note the difference between our data pre-processing and that of MiDaS)</summary>

```python
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import cv2
import torch
from torchvision.transforms import Compose

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

image = cv2.cvtColor(cv2.imread('your image path'), cv2.COLOR_BGR2RGB) / 255.0
image = transform({'image': image})['image']
image = torch.from_numpy(image).unsqueeze(0)

# depth shape: 1xHxW
depth = depth_anything(image)
```
</details>

### Do not want to define image pre-processing or download model definition files?

Easily use Depth Anything through [``transformers``](https://github.com/huggingface/transformers) within 3 lines of code! Please refer to [these instructions](https://huggingface.co/docs/transformers/main/model_doc/depth_anything) (credit to [@niels](https://huggingface.co/nielsr)).

**Note:** If you encounter ``KeyError: 'depth_anything'``, please install the latest [``transformers``](https://github.com/huggingface/transformers) from source:
```bash
pip install git+https://github.com/huggingface/transformers.git
```
<details>
<summary>Click here for a brief demo:</summary>

```python
from transformers import pipeline
from PIL import Image

image = Image.open('Your-image-path')
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
depth = pipe(image)["depth"]
```
</details>


## Citation

```bibtex
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```
