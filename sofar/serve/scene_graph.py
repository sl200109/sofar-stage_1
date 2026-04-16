import os
import numpy as np
from serve import pointso
from serve.utils import remove_outliers
from open6dor.utils import canonical_object_key, normalize_object_name, resolve_orientation_template, load_orientation_templates


_OPEN6DOR_ORIENTATION_TEMPLATES = load_orientation_templates()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _should_save_debug_artifacts(save_debug_artifacts):
    if save_debug_artifacts is None:
        return _env_flag("SOFAR_SAVE_DEBUG_ARTIFACTS", True)
    return save_debug_artifacts


def _save_npy_if_needed(path, array, save_debug_artifacts):
    if save_debug_artifacts:
        np.save(path, array)


def _filter_colored_point_cloud(colored_object_pcd):
    if colored_object_pcd.size == 0:
        return colored_object_pcd
    filtered = remove_outliers(colored_object_pcd)
    if filtered.size == 0:
        return colored_object_pcd
    return filtered


def _filter_xyz_points(segmented_object):
    if segmented_object.size == 0:
        return segmented_object
    filtered = remove_outliers(segmented_object)
    if filtered.size == 0:
        return segmented_object
    return filtered


def _resolve_open6dor_object_index(object_names, picked_object_name):
    if picked_object_name in object_names:
        return object_names.index(picked_object_name)

    canonical_target = canonical_object_key(picked_object_name)
    if not canonical_target:
        raise ValueError(f"Unable to resolve empty picked object against {object_names}")

    alias_candidates = [canonical_target]
    template = resolve_orientation_template(picked_object_name, _OPEN6DOR_ORIENTATION_TEMPLATES)
    if template:
        alias_candidates.extend(template.get("aliases", []))

    alias_fallbacks = {
        "usb": ["flash drive", "u disk", "thumb drive"],
        "mobile phone": ["phone", "smartphone", "cell phone"],
        "binder clips": ["binder clip", "clips"],
    }
    alias_candidates.extend(alias_fallbacks.get(canonical_target, []))
    alias_candidates = {canonical_object_key(value) for value in alias_candidates if str(value or "").strip()}

    for idx, name in enumerate(object_names):
        if canonical_object_key(name) in alias_candidates:
            return idx

    normalized_names = [normalize_object_name(name) for name in object_names]
    raise ValueError(f"{picked_object_name!r} is not in list after normalization. available={normalized_names}")


def build_open6dor_lightweight_scene_graph(
    picked_object_info,
    other_objects_info,
    picked_object_dict,
    target_orientation=None,
):
    target_orientation = target_orientation or {}
    return {
        "picked_object": {
            "object_name": picked_object_info.get("object name"),
            "center": picked_object_info.get("center"),
            "bounding_box": picked_object_info.get("bounding box"),
            "init_orientation": picked_object_dict.get("orientation", {}),
            "target_orientation_hint": target_orientation,
        },
        "related_objects": [
            {
                "object_name": item.get("object name"),
                "center": item.get("center"),
                "bounding_box": item.get("bounding box"),
            }
            for item in other_objects_info
        ],
    }


def open6dor_scene_graph(
    image,
    pcd,
    mask,
    info,
    object_names,
    orientation_model,
    output_folder="output",
    save_debug_artifacts=None,
):
    n, h, w = mask.shape
    image = np.array(image)
    save_debug_artifacts = _should_save_debug_artifacts(save_debug_artifacts)

    picked_object_name = info["picked_object"]
    index = _resolve_open6dor_object_index(object_names, picked_object_name)

    object_mask = mask[index]
    segmented_object = pcd[object_mask]
    segmented_image = image[object_mask]
    colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    _save_npy_if_needed(os.path.join(output_folder, "picked_obj_mask.npy"), object_mask, save_debug_artifacts)
    _save_npy_if_needed(os.path.join(output_folder, "picked_obj.npy"), colored_object_pcd, save_debug_artifacts)

    colored_object_pcd = _filter_colored_point_cloud(colored_object_pcd)
    segmented_object = colored_object_pcd[:, :3]
    min_values = segmented_object.min(axis=0)
    max_values = segmented_object.max(axis=0)
    mean_values = segmented_object.mean(axis=0)

    center = [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)]
    center_str = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
    bbox = [
        [min_values[0], max_values[0]],
        [min_values[1], max_values[1]],
        [min_values[2], max_values[2]]
    ]
    bbox_str = {
        "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
        "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
        "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
    }
    orientation = {}
    orientation_str = {}
    direction_attributes = info["direction_attributes"]
    if len(direction_attributes) > 0:
        pred = pointso.pred_orientation(orientation_model, colored_object_pcd, direction_attributes)
        for j, attribute in enumerate(direction_attributes):
            orientation[attribute] = pred[j]
            orientation_str[attribute] = f"direction vector: ({pred[j][0]:.2f}, {pred[j][1]:.2f}, {pred[j][2]:.2f})"
    picked_object_dict = {
        'object name': object_names[index],
        'center': center,
        'bounding box': bbox,
        'orientation': orientation
    }
    picked_object_info = {
        'object name': object_names[index],
        'center': center_str,
        'bounding box': bbox_str,
        # 'orientation': orientation_str
    }

    obj_id = 0
    other_objects_info = []
    for i in range(n):
        if i == index:
            continue
        else:
            obj_id += 1
        object_mask = mask[i]
        segmented_object = pcd[object_mask]
        segmented_image = image[object_mask]
        colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
        _save_npy_if_needed(
            os.path.join(output_folder, f"else_obj_{obj_id}.npy"),
            colored_object_pcd,
            save_debug_artifacts,
        )

        segmented_object = _filter_xyz_points(segmented_object)
        min_values = segmented_object.min(axis=0)
        max_values = segmented_object.max(axis=0)
        mean_values = segmented_object.mean(axis=0)
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }

        node = {
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox
        }
        other_objects_info.append(node)

    return picked_object_info, other_objects_info, picked_object_dict


def open6dor_scene_graph_for_mismatch_pcd(image, pcd, mask, info, object_names, orientation_model, output_folder="output"):
    from skimage.morphology import disk
    from scipy.ndimage import binary_dilation
    n, h, w = mask.shape
    image = np.array(image)

    picked_object_name = info["picked_object"]
    index = _resolve_open6dor_object_index(object_names, picked_object_name)

    # pcd size is not equal to mask size, use relative scale to get the mask
    object_mask = mask[index]

    radius = 10
    struct_element = disk(radius)
    object_mask = binary_dilation(object_mask, structure=struct_element)

    H, W = image.shape[:2]
    x_min, x_max = pcd[:, 0].min(), pcd[:, 0].max()
    y_min, y_max = pcd[:, 1].min(), pcd[:, 1].max()
    segmented_object = []
    mask_index = []
    for p in pcd:
        i_index = (x_max - p[0]) / (x_max - x_min) * (W - 1)
        j_index = (p[1] - y_min) / (y_max - y_min) * (H - 1)
        i_index = int(i_index)
        j_index = int(j_index)
        if object_mask[j_index, i_index]:
            segmented_object.append(list(p) + list(image[j_index, i_index]))
            mask_index.append(True)
        else:
            mask_index.append(False)
    segmented_object = np.array(segmented_object)
    np.save(os.path.join(output_folder, "picked_obj_mask.npy"), object_mask)
    np.save(os.path.join(output_folder, "picked_obj.npy"), segmented_object.reshape(-1, 6))

    segmented_object = remove_outliers(segmented_object)
    min_values = segmented_object.min(axis=0).astype(np.float64).tolist()
    max_values = segmented_object.max(axis=0).astype(np.float64).tolist()
    mean_values = segmented_object.mean(axis=0).astype(np.float64).tolist()

    center = [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)]
    center_str = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
    bbox = [
        [min_values[0], max_values[0]],
        [min_values[1], max_values[1]],
        [min_values[2], max_values[2]]
    ]
    bbox_str = {
        "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
        "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
        "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
    }
    orientation = {}
    orientation_str = {}
    direction_attributes = info["direction_attributes"]
    if len(direction_attributes) > 0:
        pred = pointso.pred_orientation(orientation_model, segmented_object.reshape(-1, 6), direction_attributes)
        for j, attribute in enumerate(direction_attributes):
            orientation[attribute] = pred[j]
            orientation_str[attribute] = f"direction vector: ({pred[j][0]:.2f}, {pred[j][1]:.2f}, {pred[j][2]:.2f})"
    picked_object_dict = {
        'object name': object_names[index],
        'center': center,
        'bounding box': bbox,
        'orientation': orientation
    }
    picked_object_info = {
        'object name': object_names[index],
        'center': center_str,
        'bounding box': bbox_str,
        # 'orientation': orientation_str
    }

    obj_id = 0
    other_objects_info = []
    for i in range(n):
        if i == index:
            continue
        else:
            obj_id += 1
        object_mask = mask[i]

        radius = 10
        struct_element = disk(radius)
        object_mask = binary_dilation(object_mask, structure=struct_element)

        H, W = image.shape[:2]
        x_min, x_max = pcd[:, 0].min(), pcd[:, 0].max()
        y_min, y_max = pcd[:, 1].min(), pcd[:, 1].max()
        segmented_object = []
        for p in pcd:
            i_index = (x_max - p[0]) / (x_max - x_min) * (W - 1)
            j_index = (p[1] - y_min) / (y_max - y_min) * (H - 1)
            i_index = int(i_index)
            j_index = int(j_index)
            if object_mask[j_index, i_index]:
                segmented_object.append(list(p) + list(image[j_index, i_index]))
        segmented_object = np.array(segmented_object)
        np.save(os.path.join(output_folder, f"else_obj_{obj_id}.npy"), segmented_object.reshape(-1, 6))

        segmented_object = remove_outliers(segmented_object)
        min_values = segmented_object.min(axis=0).astype(np.float64).tolist()
        max_values = segmented_object.max(axis=0).astype(np.float64).tolist()
        mean_values = segmented_object.mean(axis=0).astype(np.float64).tolist()
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }

        node = {
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox
        }
        other_objects_info.append(node)

    return picked_object_info, other_objects_info, picked_object_dict, np.array(mask_index)


def get_scene_graph(
    image,
    pcd,
    mask,
    info,
    object_names,
    orientation_model,
    output_folder="output",
    save_debug_artifacts=None,
):
    if len(mask) == 0:
        return [], []
    n, h, w = mask.shape
    image = np.array(image)
    save_debug_artifacts = _should_save_debug_artifacts(save_debug_artifacts)

    objects_info = []
    objects_dict = []
    for i in range(n):
        object_mask = mask[i]
        segmented_object = pcd[object_mask]
        segmented_image = image[object_mask]
        colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
        _save_npy_if_needed(
            os.path.join(output_folder, f"obj_{i + 1}.npy"),
            colored_object_pcd,
            save_debug_artifacts,
        )

        colored_object_pcd = _filter_colored_point_cloud(colored_object_pcd)
        segmented_object = colored_object_pcd[:, :3]
        min_values = segmented_object.min(axis=0)
        max_values = segmented_object.max(axis=0)
        mean_values = segmented_object.mean(axis=0)
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }
        orientation = {}
        orientation_str = {}
        direction_attributes = info[object_names[i]]
        if len(direction_attributes) > 0:
            pred = pointso.pred_orientation(orientation_model, colored_object_pcd, direction_attributes)
            for j, attribute in enumerate(direction_attributes):
                orientation[attribute] = pred[j]
                orientation_str[attribute] = f"direction vector: ({pred[j][0]:.2f}, {pred[j][1]:.2f}, {pred[j][2]:.2f})"

        node = {
            'id': i + 1,
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox,
            'orientation': orientation_str
        }
        objects_info.append(node)

        node_dict = {
            'object name': object_names[i],
            'center': [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)],
            'bounding box': [
                [min_values[0], max_values[0]],
                [min_values[1], max_values[1]],
                [min_values[2], max_values[2]]
            ],
            'orientation': orientation
        }
        objects_dict.append(node_dict)

    return objects_info, objects_dict


def get_scene_graph_3d(image, pcd, mask, info, object_names, output_folder="output", save_debug_artifacts=None):
    if len(mask) == 0:
        return [], []
    n, h, w = mask.shape
    image = np.array(image)
    save_debug_artifacts = _should_save_debug_artifacts(save_debug_artifacts)

    objects_info = []
    objects_dict = []
    for i in range(n):
        object_mask = mask[i]
        segmented_object = pcd[object_mask]
        segmented_image = image[object_mask]
        colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
        _save_npy_if_needed(
            os.path.join(output_folder, f"obj_{i + 1}.npy"),
            colored_object_pcd,
            save_debug_artifacts,
        )

        segmented_object = _filter_xyz_points(segmented_object)
        min_values = segmented_object.min(axis=0)
        max_values = segmented_object.max(axis=0)
        mean_values = segmented_object.mean(axis=0)
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }
        node = {
            'id': i + 1,
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox,
        }
        objects_info.append(node)

        node_dict = {
            'object name': object_names[i],
            'center': [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)],
            'bounding box': [
                [min_values[0], max_values[0]],
                [min_values[1], max_values[1]],
                [min_values[2], max_values[2]]
            ]
        }
        objects_dict.append(node_dict)

    return objects_info, objects_dict
