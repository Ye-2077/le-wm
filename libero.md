For Libero training and simulation evaluation, install the official stack as well:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
pip install robomimic robosuite h5py imageio imageio-ffmpeg
```

Headless MuJoCo rendering uses EGL:
```bash
export MUJOCO_GL=egl
```


### Libero data

Libero support expects the official robomimic-style task HDF5 files under `LIBERO_DATASET_ROOT`:
```bash
export LIBERO_DATASET_ROOT=/path/to/libero/datasets
```

The repository will convert those task files into a cache HDF5 compatible with `stable_worldmodel` on first use. Converted files are written under `$STABLEWM_HOME` by default, or `data.libero.cache_dir` if set.


Single-task Libero training:
```bash
python train.py data=libero_task data.libero.benchmark=libero_object \
  data.libero.task_name=pick_up_the_alphabet_soup_and_place_it_in_the_basket
```

Suite multitask Libero training:
```bash
python train.py data=libero_suite data.libero.benchmark=libero_object \
  output_model_name=lewm_libero_object
```


Libero headless simulation evaluation:
```bash
python eval.py --config-name=libero.yaml \
  policy=libero_object/lewm \
  eval.libero.benchmark=libero_object \
  eval.libero.num_episodes_per_task=10
```

To evaluate a single Libero task and save every rollout video:
```bash
python eval.py --config-name=libero.yaml \
  policy=libero_object/lewm \
  eval.libero.benchmark=libero_object \
  eval.libero.task_name=pick_up_the_alphabet_soup_and_place_it_in_the_basket \
  eval.video.save_dir=/tmp/libero_eval_videos
```

Each episode is saved as an `.mp4` file named with benchmark, task, episode index, and success flag.