# Turbulence Mitigation Transformer

[Project page](https://xg416.github.io/TMT/) | [Paper](https://arxiv.org/abs/2207.06465) | [Pre-trained Models](https://drive.google.com/drive/folders/1qKgpjH2EdZrnvEraIsMAW1Y3BiosQtvn?usp=drive_link)

Our synthetic data for the [dynamic scene](https://app.box.com/s/q6l9mcyl73r5apwwat05xlf16zf7sua4) (videos) and [static scene](https://app.box.com/s/c5wdsnxe0ax75e723jz8gk0dpai5zb7i) (image sequences) is available.

<p align="center">
  <img src="figs/video_22.gif" />
</p>

<p align="center">
  <img src="figs/pg96.gif" />
</p>

Our code was developed and tested on Ubuntu 20.04/CentOS 7 and Python 3.8.

## Quick Start
### Installation
First, clone our repo. Then,
```
cd code
pip install -r requirements. txt
```

### Training 
<details>
<summary>First, train the tilt-removal model</summary>

For the *dynamic scene modality*, run the following:
```
python train_tilt_dynamic.py --train_path ${your_training_data_path} --val_path ${your_validation_data_path} --log_path ${path_for_log_files}
```

Similarly, for the *static scene modality*, run the following:
```
python train_tilt_static.py --train_path ${your_training_data_path} --val_path ${your_validation_data_path} --log_path ${path_for_log_files}
```
</details>

<details>
<summary>Next, train the deblurring model</summary>

For the *dynamic scene modality*, run the following:
```
python train_TMT_dynamic_2stage.py --path_tilt ${your_tilt_removal_model_path} --train_path ${your_training_data_path} --val_path ${your_validation_data_path} --log_path ${path_for_log_files} --run_name ${your_exp_name}
```

Similarly, for the *static scene modality*, run the following:
```
python train_TMT_static_2stage.py --path_tilt ${your_tilt_removal_model_path} --train_path ${your_training_data_path} --val_path ${your_validation_data_path} --log_path ${path_for_log_files} --run_name ${your_exp_name}
```
</details>

<details>
<summary>Alternatively, you can directly train a one-stage model</summary>

For the *dynamic scene modality*, run the following:
```
python train_TMT_dynamic.py --train_path ${your_training_data_path} --val_path ${your_validation_data_path} --log_path ${path_for_log_files} --run_name ${your_exp_name}
```

Similarly, for the *static scene modality*, run the following:
```
python train_TMT_static.py --train_path ${your_training_data_path} --val_path ${your_validation_data_path} --log_path ${path_for_log_files} --run_name ${your_exp_name}
```
</details>

### Testing 
<details>
<summary>Test the two-stage model (tilt_removal + deblurring)</summary>

For the *dynamic scene modality*, run the following:
```
python test_TMT_dynamic_2stage.py --path_tilt ${your_tilt_removal_model_path} --model_path ${your_deblurring_model_path} --data_path ${your_validation_data_path} --result_path ${path_to_save_results}
```

Similarly, for the *static scene modality*, run the following:
```
python test_TMT_static_2stage.py --path_tilt ${your_tilt_removal_model_path} --model_path ${your_deblurring_model_path} --data_path ${your_validation_data_path} --result_path ${path_to_save_results}
```
</details>

<details>
<summary>Test the one-stage model</summary>

For the *dynamic scene modality*, run the following:
```
python test_TMT_dynamic.py --model_path ${your_model_path} --data_path ${your_validation_data_path} --result_path ${path_to_save_results}
```

Similarly, for the *static scene modality*, run the following:
```
python test_TMT_static.py --model_path ${your_model_path} --data_path ${your_validation_data_path} --result_path ${path_to_save_results}
```
</details>

All checkpoints saved during training are stored at ${log_path}/{run_name}/checkpoints. To customize your training pipeline, you can read and use the arguments in the Python scripts.
The testing may take 10-20 hours.

If you find our work helps, please consider citing our work:
```
@ARTICLE{Zhang_TMT,
  author={Zhang, Xingguang and Mao, Zhiyuan and Chimitt, Nicholas and Chan, Stanley H.},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Imaging Through the Atmosphere Using Turbulence Mitigation Transformer}, 
  year={2024},
  volume={10},
  pages={115-128},
  doi={10.1109/TCI.2024.3354421}}
```
