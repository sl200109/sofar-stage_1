import os
import glob
import json
import time
import evaluator
import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve import runtime_paths
from serve.batch_logging import setup_timestamped_logging, write_json_outputs


def evaluate_spatial_pos(task_paths, eval_file):
    eval_dict = {}
    level1 = []
    level2 = []
    all = []
    for config_file in task_paths:
        task_dir = config_file.split("/")[:-1]
        task_dir = "/".join(task_dir)
        task = task_dir.split("/")[-1]

        result_file = os.path.join(task_dir, "output/result.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
                pred_position = result["target_position"]
        else:
            pred_position = [0, 0, 0]

        with open(config_file, "r") as f:
            task_config = json.load(f)
        init_obj_pos = [p[:3] for p in task_config["init_obj_pos"]]
        pos_tag = task_config["position_tag"]
        target_position = pred_position
        if pos_tag in ["left", "right", "front", "behind", "top"]:
            sel_pos = init_obj_pos[0]
            success = evaluator.evaluate_posi(target_position, pos_tag, sel_pos)
            level1.append(success)
        elif pos_tag == "between":
            sel_pos_1 = init_obj_pos[0]
            sel_pos_2 = init_obj_pos[1]
            success = evaluator.evaluate_posi(target_position, pos_tag, sel_pos_1=sel_pos_1, sel_pos_2=sel_pos_2)
            level2.append(success)
        elif pos_tag == "center":
            sel_pos_all = init_obj_pos[:-1]
            success = evaluator.evaluate_posi(target_position, pos_tag, sel_pos_all=sel_pos_all)
            level2.append(success)
        else:
            success = 0
            print('ERROR: position tag not found')
        all.append(success)
        success = 1 if success else 0
        eval_dict[task] = {"success": success, "proposal": pred_position}
    print("level1 acc:", sum(level1) / (len(level1) + 1e-5))
    print("level2 acc:", sum(level2) / (len(level2) + 1e-5))
    print("all acc:", sum(all) / (len(all) + 1e-5))

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=4, ensure_ascii=False)
    return eval_dict


def evaluate_spatial_rot(task_paths, eval_file):
    eval_dict = {}
    level1 = []
    level2 = []
    level3 = []
    all = []
    for config_file in task_paths:

        task_dir = config_file.split("/")[:-1]
        task_dir = "/".join(task_dir)
        task = task_dir.split("/")[-1]

        with open(config_file, "r") as f:
            task_config = json.load(f)
        rot_tag_level = task_config["rot_tag_level"]

        result_file = os.path.join(task_dir, "output/result.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
                pred_rotation = result["transform_matrix"]
        else:
            pred_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        try:
            pred_quaternion = R.from_matrix(pred_rotation).as_quat()
        except:
            pred_quaternion = np.array([1, 0, 0, 0])

        if np.isnan(pred_quaternion).any():
            pred_quaternion = np.array([1, 0, 0, 0])
        deviation = evaluator.evaluate_rot(config_file, pred_quaternion)
        pred_quaternion = pred_quaternion.tolist()
        eval_dict[task] = {"deviation": deviation, "proposal": pred_quaternion}

        if deviation == "No annotation found" or deviation == "Annotation stage 2":
            continue
        else:
            int(deviation)

        if int(deviation) <= 45:
            if rot_tag_level == 0:
                level1.append(1)
            if rot_tag_level == 1:
                level2.append(1)
            if rot_tag_level == 2:
                level3.append(1)
            all.append(1)
        else:
            if rot_tag_level == 0:
                level1.append(0)
            if rot_tag_level == 1:
                level2.append(0)
            if rot_tag_level == 2:
                level3.append(0)
            all.append(0)
    print("level1 acc:", sum(level1) / (len(level1) + 1e-5))
    print("level2 acc:", sum(level2) / (len(level2) + 1e-5))
    print("level3 acc:", sum(level3) / (len(level3) + 1e-5))
    print("all acc:", sum(all) / (len(all) + 1e-5))

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=4, ensure_ascii=False)
    return eval_dict


def evaluate_spatial_6dof(task_paths, eval_file):
    eval_dict = {}
    pos = []
    rot = []
    all = []
    for config_file in task_paths:

        task_dir = config_file.split("/")[:-1]
        task_dir = "/".join(task_dir)
        task = task_dir.split("/")[-1]

        with open(config_file, "r") as f:
            task_config = json.load(f)

        result_file = os.path.join(task_dir, "output/result.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
                pred_rotation = result["transform_matrix"]
                pred_position = result["target_position"]
        else:
            pred_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            pred_position = [0, 0, 0]

        try:
            pred_quaternion = R.from_matrix(pred_rotation).as_quat()
        except:
            pred_quaternion = np.array([1, 0, 0, 0])

        if np.isnan(pred_quaternion).any():
            pred_quaternion = np.array([1, 0, 0, 0])
        deviation = evaluator.evaluate_rot(config_file, pred_quaternion)
        pred_quaternion = pred_quaternion.tolist()
        eval_dict[task] = {"deviation": deviation, "proposal": pred_quaternion}

        init_obj_pos = [p[:3] for p in task_config["init_obj_pos"]]
        pos_tag = task_config["position_tag"]
        target_position = pred_position
        if pos_tag in ["left", "right", "front", "behind", "top"]:
            sel_pos = init_obj_pos[0]
            success = evaluator.evaluate_posi(target_position, pos_tag, sel_pos)
            pos.append(success)
        elif pos_tag == "between":
            sel_pos_1 = init_obj_pos[0]
            sel_pos_2 = init_obj_pos[1]
            success = evaluator.evaluate_posi(target_position, pos_tag, sel_pos_1=sel_pos_1, sel_pos_2=sel_pos_2)
            pos.append(success)
        elif pos_tag == "center":
            sel_pos_all = init_obj_pos[:-1]
            success = evaluator.evaluate_posi(target_position, pos_tag, sel_pos_all=sel_pos_all)
            pos.append(success)
        else:
            sel_pos = init_obj_pos[0]
            success = evaluator.evaluate_posi(target_position, pos_tag, sel_pos)
            pos.append(success)
        success = 1 if success else 0

        eval_dict[task] = {"pos_success": success, "pred_position": pred_position, "deviation": deviation,
                           "pred_quaternion": pred_quaternion}

        if deviation == "No annotation found" or deviation == "Annotation stage 2":
            continue
        else:
            int(deviation)

        if int(deviation) <= 45:
            rot.append(1)
        else:
            rot.append(0)

        if success and int(deviation) <= 45:
            all.append(1)
        else:
            all.append(0)
    print("6-dof pos acc:", sum(pos) / (len(pos) + 1e-5))
    print("6-dof rot acc:", sum(rot) / (len(rot) + 1e-5))
    print("6-dof all acc:", sum(all) / (len(all) + 1e-5))

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=4, ensure_ascii=False)
    return eval_dict


if __name__ == "__main__":
    output_dir = runtime_paths.ensure_output_dir()
    run_id, _ = setup_timestamped_logging(output_dir, "eval_open6dor")
    dataset_root = runtime_paths.open6dor_dataset_dir()

    section_start = time.perf_counter()
    task_paths = sorted(glob.glob(str(dataset_root / 'task_refine_pos' / '*' / '*' / '*' / 'task_config_new5.json')))
    print(f"[eval_open6dor] evaluating position track with {len(task_paths)} tasks")
    output = output_dir / "eval_pos.json"
    pos_result = evaluate_spatial_pos(task_paths, str(output))
    write_json_outputs(pos_result, output_dir, "eval_pos.json", run_id)
    print(f"[eval_open6dor] position track finished in {time.perf_counter() - section_start:.2f}s")

    section_start = time.perf_counter()
    task_paths = sorted(glob.glob(str(dataset_root / 'task_refine_rot' / '*' / '*' / '*' / 'task_config_new5.json')))
    print(f"[eval_open6dor] evaluating rotation track with {len(task_paths)} tasks")
    output = output_dir / "eval_rot.json"
    rot_result = evaluate_spatial_rot(task_paths, str(output))
    write_json_outputs(rot_result, output_dir, "eval_rot.json", run_id)
    print(f"[eval_open6dor] rotation track finished in {time.perf_counter() - section_start:.2f}s")

    section_start = time.perf_counter()
    task_paths = sorted(glob.glob(str(dataset_root / 'task_refine_6dof' / '*' / '*' / '*' / 'task_config_new5.json')))
    print(f"[eval_open6dor] evaluating 6-dof track with {len(task_paths)} tasks")
    output = output_dir / "eval_6dof.json"
    dof_result = evaluate_spatial_6dof(task_paths, str(output))
    write_json_outputs(dof_result, output_dir, "eval_6dof.json", run_id)
    print(f"[eval_open6dor] 6-dof track finished in {time.perf_counter() - section_start:.2f}s")
