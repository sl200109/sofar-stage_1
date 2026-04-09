import json
import re
from pathlib import Path

from PIL import Image, ImageEnhance


ROOT_DIR = Path(__file__).resolve().parents[1]
ORIENTATION_TEMPLATE_PATH = ROOT_DIR / "open6dor" / "orientation_templates.json"


def preprocess_open6dor_image(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.5)
    width, height = image.size
    edge_width = int(width * 0.1)
    edge_height = int(height * 0.1)
    white_image = Image.new("RGB", (width, height), "white")
    white_image.paste(image.crop((edge_width, edge_height, width - edge_width, height - edge_height * 2)),
                      (edge_width, edge_height))
    return white_image


def normalize_object_name(name):
    return str(name or "").strip().replace("_", " ")


def canonical_object_key(name):
    normalized = normalize_object_name(name).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def strip_rotation_clause(instruction):
    text = str(instruction or "").strip()
    marker = "We also need to specify the rotation of the object after placement:"
    if marker in text:
        text = text.split(marker, 1)[0].strip()
    return text


def load_orientation_templates(path=None):
    template_path = Path(path) if path is not None else ORIENTATION_TEMPLATE_PATH
    with template_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    templates = {}
    for key, value in raw.items():
        canonical = canonical_object_key(key)
        aliases = value.get("aliases", [])
        direction_attributes = value.get("direction_attributes", [])
        templates[canonical] = {
            "direction_attributes": direction_attributes,
            "aliases": [canonical_object_key(alias) for alias in aliases] + [canonical],
            "description": value.get("description", ""),
        }
    return templates


def resolve_orientation_template(object_name, templates):
    canonical = canonical_object_key(object_name)
    if canonical in templates:
        return templates[canonical]
    for value in templates.values():
        if canonical in value.get("aliases", []):
            return value
    return None


def build_orientation_template_hints(object_name, templates):
    template = resolve_orientation_template(object_name, templates)
    if template is None:
        return {
            "matched": False,
            "object_name": normalize_object_name(object_name),
            "direction_attributes": [],
            "description": "",
        }
    return {
        "matched": True,
        "object_name": normalize_object_name(object_name),
        "direction_attributes": template.get("direction_attributes", []),
        "description": template.get("description", ""),
    }


def _extract_simple_reference(text):
    pattern = (
        r"\b(?:behind|in front of|left of|right of|to the left of|to the right of|"
        r"near|next to|beside)\s+(?:the\s+)?(.+?)\s+on\s+the\s+table\b"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return []
    return [normalize_object_name(match.group(1))]


def extract_open6dor_minimal_object_set(instruction, target_obj_name=""):
    base_instruction = strip_rotation_clause(instruction)
    picked_object = normalize_object_name(target_obj_name)
    relation_type = "none"
    fallback_required = False

    if not picked_object:
        match = re.search(
            r"^\s*(?:place|pick)\s+(?:the\s+)?(.+?)\s+"
            r"(?:behind|in front of|left of|right of|between|at the center of|center of|near|next to|beside|on)\b",
            base_instruction,
            flags=re.IGNORECASE,
        )
        if match:
            picked_object = normalize_object_name(match.group(1))

    lower_text = base_instruction.lower()
    related_objects = []

    between_match = re.search(
        r"\bbetween\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\s+on\s+the\s+table\b",
        base_instruction,
        flags=re.IGNORECASE,
    )
    if between_match:
        relation_type = "between"
        related_objects = [
            normalize_object_name(between_match.group(1)),
            normalize_object_name(between_match.group(2)),
        ]
    elif "center of all the objects" in lower_text or "center of all objects" in lower_text:
        relation_type = "center"
        fallback_required = True
    else:
        simple_refs = _extract_simple_reference(base_instruction)
        if simple_refs:
            relation_type = "relative"
            related_objects = simple_refs

    related_objects = [obj for obj in related_objects if obj and canonical_object_key(obj) != canonical_object_key(picked_object)]
    if relation_type not in {"between", "center"}:
        related_objects = related_objects[:2]

    return {
        "picked_object": picked_object,
        "related_objects": related_objects,
        "relation_type": relation_type,
        "fallback_required": fallback_required or not picked_object,
        "base_instruction": base_instruction,
    }
