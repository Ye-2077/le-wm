import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_100",
)


def optional_import(module_name: str, install_hint: str):
    """按需导入可选依赖，并在缺失时给出明确安装提示。"""
    try:
        module = __import__(module_name, fromlist=["__name__"])
    except ImportError as exc:
        raise ImportError(f"Missing optional dependency '{module_name}'. {install_hint}") from exc
    return module


def get_cache_dir(cache_dir: str | os.PathLike | None = None) -> Path:
    """解析 StableWM 缓存目录，优先使用显式配置，其次读取环境变量。"""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    env_dir = os.environ.get("STABLEWM_HOME")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.home() / ".stable-wm"


def normalize_task_name(name: str) -> str:
    """将任务名标准化为适合文件名和配置匹配的形式。"""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _as_list(value):
    """把单值或空值统一转成列表，便于后续按 key 遍历。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first_present(group, candidates):
    """返回候选 key 中第一个存在于 group 的名字。"""
    for key in candidates:
        if key in group:
            return key
    return None


def _resolve_task_file(task_name: str, dataset_root: Path, explicit_map: dict[str, Any] | None = None) -> Path:
    """为单个 Libero task 找到对应的原始 HDF5 文件。"""
    explicit_map = explicit_map or {}
    if task_name in explicit_map:
        return Path(explicit_map[task_name]).expanduser().resolve()

    norm_task = normalize_task_name(task_name)
    matches = []
    for path in dataset_root.rglob("*.hdf5"):
        stem = normalize_task_name(path.stem)
        if stem == norm_task or norm_task in stem or stem in norm_task:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Could not find a Libero dataset file for task '{task_name}' under {dataset_root}")
    matches.sort()
    return matches[0]


def _resolve_suite_files(
    benchmark_name: str,
    dataset_root: Path,
    task_files: dict[str, Any] | None = None,
):
    """解析一个 suite 下所有任务对应的原始 HDF5 文件路径。"""
    tasks = get_libero_suite_tasks(benchmark_name)
    resolved = []
    for task_id, task in enumerate(tasks):
        path = _resolve_task_file(task["name"], dataset_root, explicit_map=task_files)
        resolved.append(
            {
                "task_id": task_id,
                "task_name": task["name"],
                "path": path,
                "language": task.get("language"),
            }
        )
    return resolved


def get_libero_suite_tasks(benchmark_name: str) -> list[dict[str, Any]]:
    """从官方 LIBERO benchmark API 中枚举指定 suite 的任务元信息。"""
    benchmark = optional_import(
        "libero.libero.benchmark",
        "Install LIBERO with `pip install -e .` inside the official repository.",
    )
    benchmark_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        raise KeyError(f"Unknown Libero benchmark '{benchmark_name}'. Available: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[benchmark_name]()
    tasks = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        tasks.append(
            {
                "task_id": task_id,
                "name": task.name,
                "language": getattr(task, "language", ""),
                "problem_folder": getattr(task, "problem_folder", ""),
                "bddl_file": getattr(task, "bddl_file", ""),
            }
        )
    return tasks


def get_libero_task_suite(benchmark_name: str):
    """构造官方 LIBERO benchmark suite 对象，供仿真评测直接使用。"""
    benchmark = optional_import(
        "libero.libero.benchmark",
        "Install LIBERO with `pip install -e .` inside the official repository.",
    )
    benchmark_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        raise KeyError(f"Unknown Libero benchmark '{benchmark_name}'. Available: {sorted(benchmark_dict)}")
    return benchmark_dict[benchmark_name]()


def get_libero_bddl_path(task) -> Path:
    """根据 Libero task 元信息定位对应的 BDDL 文件。"""
    libero_utils = optional_import(
        "libero.libero",
        "Install LIBERO with `pip install -e .` inside the official repository.",
    )
    return Path(
        libero_utils.get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )


def discover_libero_sources(cfg_libero) -> list[dict[str, Any]]:
    """根据训练配置解析需要转换的 Libero 原始数据源列表。"""
    dataset_root = Path(cfg_libero.dataset_root).expanduser().resolve()
    task_files = dict(getattr(cfg_libero, "task_files", {}) or {})

    if cfg_libero.mode == "single_task":
        task_name = cfg_libero.task_name
        if not task_name:
            raise ValueError("`data.libero.task_name` is required for single_task mode.")
        path = _resolve_task_file(task_name, dataset_root, explicit_map=task_files)
        return [{"task_id": 0, "task_name": task_name, "path": path}]

    if cfg_libero.benchmark == "all":
        sources = []
        next_task_id = 0
        for suite_name in LIBERO_SUITES:
            suite_sources = _resolve_suite_files(suite_name, dataset_root, task_files=task_files)
            for item in suite_sources:
                item["task_id"] = next_task_id
                item["benchmark"] = suite_name
                next_task_id += 1
            sources.extend(suite_sources)
        return sources

    return _resolve_suite_files(cfg_libero.benchmark, dataset_root, task_files=task_files)


def _collect_lowdim(obs_group, include_keys: list[str] | None = None, exclude_keys: list[str] | None = None):
    """从 robomimic obs group 中收集并拼接低维观测。"""
    include_keys = include_keys or []
    exclude_keys = set(exclude_keys or [])

    if include_keys:
        arrays = [np.asarray(obs_group[key]) for key in include_keys if key in obs_group]
        if arrays:
            return np.concatenate([arr.reshape(arr.shape[0], -1) for arr in arrays], axis=-1)
        return None

    arrays = []
    for key in obs_group.keys():
        if key in exclude_keys:
            continue
        arr = np.asarray(obs_group[key])
        if arr.ndim >= 3:
            continue
        arrays.append(arr.reshape(arr.shape[0], -1))
    if not arrays:
        return None
    return np.concatenate(arrays, axis=-1)


def _read_demo_arrays(demo_group, obs_key: str, proprio_keys=None, state_keys=None):
    """读取单条 demo，并抽取图像、动作以及可选低维状态。"""
    obs_group = demo_group["obs"]
    image_key = obs_key if obs_key in obs_group else _first_present(
        obs_group,
        [
            "agentview_rgb",
            "robot0_eye_in_hand_image",
            "robot0_agentview_left_image",
            "rgb",
            "image",
            "pixels",
        ],
    )
    if image_key is None:
        raise KeyError(f"Could not find an image observation key. Available keys: {list(obs_group.keys())}")

    pixels = np.asarray(obs_group[image_key])
    if pixels.ndim != 4:
        raise ValueError(f"Expected image observations with shape (T, H, W, C), got {pixels.shape}")

    action_key = "actions" if "actions" in demo_group else "action"
    actions = np.asarray(demo_group[action_key])
    proprio = _collect_lowdim(obs_group, include_keys=_as_list(proprio_keys), exclude_keys=[image_key]) if proprio_keys is not None else _collect_lowdim(obs_group, exclude_keys=[image_key])
    state = _collect_lowdim(obs_group, include_keys=_as_list(state_keys), exclude_keys=[image_key]) if state_keys is not None else proprio

    return {
        "pixels": pixels,
        "action": actions,
        "proprio": proprio,
        "state": state,
        "obs_key": image_key,
    }


def _init_storage():
    """初始化转换阶段的列式缓存容器。"""
    return {
        "pixels": [],
        "action": [],
        "proprio": [],
        "state": [],
        "ep_idx": [],
        "step_idx": [],
        "task_id": [],
        "task_name": [],
        "ep_len": [],
    }


def _append_episode(storage, arrays, task_name: str, task_id: int, episode_idx: int):
    """将单条 episode 追加到统一缓存结构中。"""
    n_steps = arrays["action"].shape[0]
    pixels = arrays["pixels"][:n_steps]

    storage["pixels"].append(np.asarray(pixels, dtype=np.uint8))
    storage["action"].append(np.asarray(arrays["action"], dtype=np.float32))
    storage["ep_idx"].append(np.full((n_steps,), episode_idx, dtype=np.int32))
    storage["step_idx"].append(np.arange(n_steps, dtype=np.int32))
    storage["task_id"].append(np.full((n_steps,), task_id, dtype=np.int32))
    storage["task_name"].append(np.asarray([task_name] * n_steps, dtype=object))
    storage["ep_len"].append(np.asarray([n_steps], dtype=np.int32))

    for key in ("proprio", "state"):
        value = arrays[key]
        if value is not None:
            storage[key].append(np.asarray(value[:n_steps], dtype=np.float32))


def _concat_storage(storage):
    """将逐 episode 暂存的数据按列拼接成可直接写盘的数组。"""
    out = {}
    for key, parts in storage.items():
        if not parts:
            continue
        if key == "task_name":
            out[key] = np.concatenate(parts)
        else:
            out[key] = np.concatenate(parts, axis=0)
    if "ep_len" in out:
        lengths = out["ep_len"].astype(np.int32)
        out["ep_len"] = lengths
        out["ep_offset"] = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
        )
    return out


def convert_libero_to_hdf5(
    sources: list[dict[str, Any]],
    output_path: Path,
    obs_key: str,
    force: bool = False,
    proprio_keys=None,
    state_keys=None,
):
    """把官方 Libero / robomimic HDF5 转成 stable_worldmodel 可读的缓存格式。"""
    h5py = optional_import("h5py", "Install h5py to enable Libero dataset conversion.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        return output_path

    storage = _init_storage()
    episode_idx = 0
    obs_key_used = None

    for source in sources:
        source_path = Path(source["path"]).expanduser().resolve()
        with h5py.File(source_path, "r") as handle:
            demos_group = handle["data"] if "data" in handle else handle
            demo_names = sorted(demos_group.keys())
            for demo_name in demo_names:
                demo_group = demos_group[demo_name]
                arrays = _read_demo_arrays(
                    demo_group,
                    obs_key=obs_key,
                    proprio_keys=proprio_keys,
                    state_keys=state_keys,
                )
                obs_key_used = arrays["obs_key"]
                _append_episode(
                    storage,
                    arrays,
                    task_name=source["task_name"],
                    task_id=source["task_id"],
                    episode_idx=episode_idx,
                )
                episode_idx += 1

    merged = _concat_storage(storage)
    with h5py.File(output_path, "w") as handle:
        for key, value in merged.items():
            if key == "task_name":
                dtype = h5py.string_dtype(encoding="utf-8")
                utf8_values = [
                    item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
                    for item in value.tolist()
                ]
                handle.create_dataset(key, data=utf8_values, dtype=dtype)
                continue
            handle.create_dataset(key, data=value)
        handle.attrs["format"] = "lewm_libero_v1"
        handle.attrs["num_episodes"] = episode_idx
        if obs_key_used is not None:
            handle.attrs["obs_key"] = obs_key_used
    return output_path


def ensure_libero_cache(cfg_libero) -> tuple[str, Path]:
    """确保训练所需的 Libero 缓存数据存在，不存在时自动构建。"""
    cache_dir = get_cache_dir(getattr(cfg_libero, "cache_dir", None))
    cache_name = getattr(cfg_libero, "cache_name", None)
    if not cache_name:
        if cfg_libero.mode == "single_task":
            cache_name = f"libero_{normalize_task_name(cfg_libero.task_name)}"
        else:
            cache_name = f"{cfg_libero.benchmark}_multitask"

    output_path = cache_dir / f"{cache_name}.h5"
    sources = discover_libero_sources(cfg_libero)
    convert_libero_to_hdf5(
        sources=sources,
        output_path=output_path,
        obs_key=cfg_libero.obs_key,
        force=bool(getattr(cfg_libero, "force_rebuild", False)),
        proprio_keys=getattr(cfg_libero, "proprio_keys", None),
        state_keys=getattr(cfg_libero, "state_keys", None),
    )
    return cache_name, output_path


def load_goal_image_from_file(task_path: Path, obs_key: str) -> np.ndarray:
    """从任务数据中取一张目标图像，作为规划时的 goal observation。"""
    h5py = optional_import("h5py", "Install h5py to enable Libero dataset loading.")
    with h5py.File(task_path, "r") as handle:
        demos_group = handle["data"] if "data" in handle else handle
        demo_name = sorted(demos_group.keys())[0]
        arrays = _read_demo_arrays(demos_group[demo_name], obs_key=obs_key)
    return arrays["pixels"][-1]


def fit_libero_action_scaler(task_paths: list[Path]):
    """用评测任务的数据拟合动作标准化器，保证 planner 动作尺度与训练一致。"""
    preprocessing = optional_import(
        "sklearn.preprocessing",
        "Install scikit-learn to fit Libero action normalization statistics.",
    )
    h5py = optional_import("h5py", "Install h5py to enable Libero dataset loading.")
    scaler = preprocessing.StandardScaler()
    actions = []
    for task_path in task_paths:
        with h5py.File(task_path, "r") as handle:
            demos_group = handle["data"] if "data" in handle else handle
            for demo_name in sorted(demos_group.keys()):
                demo_group = demos_group[demo_name]
                action_key = "actions" if "actions" in demo_group else "action"
                actions.append(np.asarray(demo_group[action_key]))
    if not actions:
        raise ValueError("Could not fit Libero action scaler because no action data was found.")
    scaler.fit(np.concatenate(actions, axis=0))
    return scaler


def extract_pixels_from_obs(obs: Any, obs_key: str):
    """从环境观测中尽量稳健地提取图像帧。"""
    if isinstance(obs, dict):
        if obs_key in obs:
            return np.asarray(obs[obs_key])
        if "pixels" in obs:
            return np.asarray(obs["pixels"])
        for key, value in obs.items():
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                return arr
    arr = np.asarray(obs)
    if arr.ndim == 3:
        return arr
    raise KeyError(f"Could not extract image observation using key '{obs_key}'.")


def frame_from_env(env, obs: Any, obs_key: str):
    """优先从 observation 取图像，失败时回退到 env.render()。"""
    try:
        frame = extract_pixels_from_obs(obs, obs_key)
        if frame.ndim == 3:
            return frame
    except Exception:
        pass

    render = getattr(env, "render", None)
    if callable(render):
        frame = render()
        frame = np.asarray(frame)
        if frame.ndim == 3:
            return frame
    raise RuntimeError("Could not extract frames from Libero env. Ensure the env exposes image obs or render().")


@dataclass
class SingleEnvAdapter:
    """把单环境包装成 stable_worldmodel policy 期望的最小 env 接口。"""
    env: Any

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def num_envs(self):
        return 1


class LiberoPolicyAdapter:
    """把 WorldModelPolicy 适配到 Libero 单环境 rollout。"""

    def __init__(self, policy, history_len: int, goal_image: np.ndarray, obs_key: str):
        self.policy = policy
        self.history_len = history_len
        self.goal_image = np.asarray(goal_image)
        self.obs_key = obs_key
        self.history = deque(maxlen=history_len)

    def reset(self, env, init_obs):
        """重置历史帧缓存，并把策略绑定到当前环境。"""
        self.history.clear()
        first_frame = extract_pixels_from_obs(init_obs, obs_key=self.obs_key)
        for _ in range(self.history_len):
            self.history.append(first_frame)
        self.policy.set_env(SingleEnvAdapter(env))

    def get_action(self, current_frame: np.ndarray):
        """组装 JEPA 规划所需的 history + goal 输入，并返回当前动作。"""
        self.history.append(np.asarray(current_frame))
        info = {
            "pixels": np.expand_dims(np.stack(list(self.history), axis=0), axis=0),
            "goal": np.expand_dims(np.expand_dims(self.goal_image, axis=0), axis=0),
        }
        action = self.policy.get_action(info)
        action = np.asarray(action)
        if action.ndim > 1:
            action = action[0]
        return action


def make_libero_policy(cfg, process, transform):
    """按配置构造用于 Libero 评测的 world-model policy。"""
    if cfg.get("policy", "random") == "random":
        return None

    swm = optional_import(
        "stable_worldmodel",
        "Install stable-worldmodel[env,train] before running evaluation.",
    )
    hydra = optional_import("hydra", "Install hydra-core to instantiate evaluation solvers.")

    policy_name = cfg.policy
    model = swm.policy.AutoCostModel(policy_name)
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.policy.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )


def compute_success(done: bool, reward: float, info: dict):
    """从 Libero / Gym 风格 step 返回值中提取 success 标志。"""
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success"):
            if key in info:
                value = info[key]
                if isinstance(value, np.ndarray):
                    return bool(np.any(value))
                return bool(value)
    return bool(done or reward > 0)


def record_episode_video(video_path: Path, frames: list[np.ndarray], fps: int):
    """把单条 episode 的帧序列编码为 mp4 视频。"""
    if not frames:
        return
    imageio = optional_import("imageio", "Install imageio to save Libero rollout videos.")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def evaluate_libero(cfg, process, transform):
    """在官方 Libero 仿真环境中执行 headless rollout，并输出任务级指标与视频。"""
    os.environ.setdefault("MUJOCO_GL", "egl")

    libero_envs = optional_import(
        "libero.libero.envs",
        "Install LIBERO with its environment dependencies before running Libero eval.",
    )

    results_dir = Path(cfg.eval.video.save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    suite = get_libero_task_suite(cfg.eval.libero.benchmark)
    dataset_root = Path(cfg.eval.libero.dataset_root).expanduser().resolve()
    task_file_map = dict(getattr(cfg.eval.libero, "task_files", {}) or {})
    selected_tasks = []

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        if cfg.eval.libero.task_name and task.name != cfg.eval.libero.task_name:
            continue
        selected_tasks.append((task_id, task))

    if not selected_tasks:
        raise ValueError("No Libero tasks selected for evaluation.")

    task_paths = [_resolve_task_file(task.name, dataset_root, explicit_map=task_file_map) for _, task in selected_tasks]
    process = dict(process)
    if cfg.policy != "random":
        process["action"] = fit_libero_action_scaler(task_paths)

    metrics_by_task = {}
    all_success = []
    all_lengths = []

    for task_id, task in selected_tasks:
        task_file = _resolve_task_file(task.name, dataset_root, explicit_map=task_file_map)
        goal_image = load_goal_image_from_file(task_file, cfg.eval.libero.obs_key)
        policy = make_libero_policy(cfg, process=process, transform=transform)
        init_states = suite.get_task_init_states(task_id)
        env = libero_envs.OffScreenRenderEnv(
            bddl_file_name=str(get_libero_bddl_path(task)),
            camera_heights=cfg.eval.img_size,
            camera_widths=cfg.eval.img_size,
        )
        try:
            task_success = []
            task_lengths = []
            policy_adapter = None if cfg.policy == "random" else LiberoPolicyAdapter(
                policy=policy,
                history_len=getattr(cfg.plan_config, "history_len", 3),
                goal_image=goal_image,
                obs_key=cfg.eval.libero.obs_key,
            )

            for episode_idx in range(cfg.eval.libero.num_episodes_per_task):
                env.seed(cfg.seed + episode_idx)
                reset_out = env.reset()
                init_state = init_states[episode_idx % len(init_states)]
                maybe_obs = env.set_init_state(init_state)
                obs = maybe_obs if maybe_obs is not None else reset_out

                if isinstance(obs, tuple):
                    obs = obs[0]

                current_frame = frame_from_env(env, obs, cfg.eval.libero.obs_key)
                if policy_adapter is not None:
                    policy_adapter.reset(env, current_frame)

                frames = [current_frame]
                done = False
                reward = 0.0
                info = {}
                step_count = 0

                while step_count < cfg.eval.libero.max_steps:
                    if cfg.policy == "random":
                        action = env.action_space.sample()
                    else:
                        # 规划策略按历史图像和目标图像生成下一步动作。
                        action = policy_adapter.get_action(current_frame)

                    step_out = env.step(action)
                    if len(step_out) == 4:
                        obs, reward, done, info = step_out
                    else:
                        obs, reward, terminated, truncated, info = step_out
                        done = bool(terminated or truncated)

                    current_frame = frame_from_env(env, obs, cfg.eval.libero.obs_key)
                    frames.append(current_frame)
                    step_count += 1

                    if done:
                        break

                success = compute_success(done=done, reward=reward, info=info)
                task_success.append(float(success))
                task_lengths.append(step_count)
                all_success.append(float(success))
                all_lengths.append(step_count)

                if cfg.eval.video.save_every_episode:
                    # 每个 episode 单独落盘，便于按任务和成功标记回溯行为。
                    filename = (
                        f"{cfg.eval.libero.benchmark}__{normalize_task_name(task.name)}"
                        f"__ep_{episode_idx:03d}__success_{int(success)}.mp4"
                    )
                    record_episode_video(results_dir / filename, frames=frames, fps=cfg.eval.video.fps)

            metrics_by_task[task.name] = {
                "success_rate": float(np.mean(task_success)) if task_success else 0.0,
                "avg_episode_length": float(np.mean(task_lengths)) if task_lengths else 0.0,
                "num_episodes": len(task_success),
            }
        finally:
            env.close()

    return {
        "success_rate": float(np.mean(all_success)) if all_success else 0.0,
        "avg_episode_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "num_episodes": len(all_success),
        "tasks": metrics_by_task,
        "video_dir": str(results_dir),
    }
