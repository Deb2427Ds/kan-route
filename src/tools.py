"""
tools.py — Tool taxonomy and descriptions for KAN-Route.

50 computer vision tools organized into 8 macro-categories.
"""

# ── Tool list (index = tool_id) ───────────────────────────────────────────────
TOOLS = [
    # Detection & Localization (0-8)
    "object_detection",
    "object_counting",
    "person_detection",
    "face_detection",
    "vehicle_detection",
    "animal_detection",
    "text_detection_ocr",
    "logo_detection",
    "anomaly_detection",

    # Segmentation (9-14)
    "semantic_segmentation",
    "instance_segmentation",
    "panoptic_segmentation",
    "background_removal",
    "salient_object_segmentation",
    "matting",

    # Image Generation & Editing (15-22)
    "text_to_image_generation",
    "image_inpainting",
    "image_outpainting",
    "style_transfer",
    "super_resolution",
    "image_colorization",
    "image_denoising",
    "image_deblurring",

    # 3D & Geometry (23-27)
    "depth_estimation",
    "surface_normal_estimation",
    "3d_reconstruction",
    "point_cloud_generation",
    "stereo_matching",

    # Classification & Recognition (28-35)
    "image_classification",
    "fine_grained_classification",
    "scene_recognition",
    "facial_expression_recognition",
    "age_gender_estimation",
    "action_recognition",
    "pose_estimation",
    "gaze_estimation",

    # Visual Understanding & QA (36-42)
    "visual_question_answering",
    "image_captioning",
    "visual_reasoning",
    "visual_grounding",
    "relation_detection",
    "attribute_recognition",
    "change_detection",

    # Video & Temporal (43-46)
    "video_object_tracking",
    "optical_flow_estimation",
    "video_captioning",
    "temporal_action_localization",

    # Medical & Specialized (47-49)
    "medical_image_segmentation",
    "document_layout_analysis",
    "satellite_image_analysis",
]

assert len(TOOLS) == 50, f"Expected 50 tools, got {len(TOOLS)}"

TOOL2ID = {t: i for i, t in enumerate(TOOLS)}
NUM_TOOLS = len(TOOLS)

# ── One-line descriptions (used in teacher LLM prompts) ──────────────────────
TOOL_DESCRIPTIONS = {
    "object_detection":               "Detect and draw bounding boxes around all objects",
    "object_counting":                "Count how many instances of an object appear",
    "person_detection":               "Detect and locate all people in the image",
    "face_detection":                 "Find and mark all faces",
    "vehicle_detection":              "Detect cars, trucks, bikes, and other vehicles",
    "animal_detection":               "Detect and identify animals",
    "text_detection_ocr":             "Read and extract text visible in the image",
    "logo_detection":                 "Identify brand logos or signs",
    "anomaly_detection":              "Find unusual or out-of-place objects",
    "semantic_segmentation":          "Label every pixel with its object class",
    "instance_segmentation":          "Segment each individual object instance",
    "panoptic_segmentation":          "Combine semantic and instance segmentation",
    "background_removal":             "Remove the background and isolate the foreground",
    "salient_object_segmentation":    "Highlight the most visually prominent object",
    "matting":                        "Precise foreground extraction with alpha channel",
    "text_to_image_generation":       "Generate a new image from a text description",
    "image_inpainting":               "Fill in or replace a masked region of the image",
    "image_outpainting":              "Extend the image beyond its original borders",
    "style_transfer":                 "Apply artistic style to the image",
    "super_resolution":               "Increase image resolution and sharpness",
    "image_colorization":             "Add color to a grayscale image",
    "image_denoising":                "Remove noise from the image",
    "image_deblurring":               "Sharpen a blurry image",
    "depth_estimation":               "Estimate how far each object is from the camera",
    "surface_normal_estimation":      "Compute the 3D surface orientation at each point",
    "3d_reconstruction":              "Build a 3D model from the image",
    "point_cloud_generation":         "Generate a point cloud from the scene",
    "stereo_matching":                "Compute disparity between stereo image pair",
    "image_classification":           "Identify the main category of the image",
    "fine_grained_classification":    "Identify subcategory (e.g., bird species)",
    "scene_recognition":              "Recognize the type of scene or environment",
    "facial_expression_recognition":  "Identify the emotion shown on a face",
    "age_gender_estimation":          "Estimate age and gender of a person",
    "action_recognition":             "Identify what action is being performed",
    "pose_estimation":                "Estimate the body keypoints and skeleton",
    "gaze_estimation":                "Determine where a person is looking",
    "visual_question_answering":      "Answer a natural language question about the image",
    "image_captioning":               "Generate a descriptive sentence for the image",
    "visual_reasoning":               "Perform multi-step reasoning about image content",
    "visual_grounding":               "Locate a specific object described in text",
    "relation_detection":             "Identify relationships between objects",
    "attribute_recognition":          "Recognize properties like color, size, texture",
    "change_detection":               "Find differences between two images",
    "video_object_tracking":          "Track an object across video frames",
    "optical_flow_estimation":        "Compute pixel-level motion between frames",
    "video_captioning":               "Generate a description for a video clip",
    "temporal_action_localization":   "Locate when an action happens in a video",
    "medical_image_segmentation":     "Segment anatomical structures in medical scans",
    "document_layout_analysis":       "Parse the structure of a document image",
    "satellite_image_analysis":       "Analyze and classify satellite/aerial imagery",
}

assert len(TOOL_DESCRIPTIONS) == 50

# ── Tool batches for LLM-based generation (10 tools per batch) ───────────────
TOOL_BATCHES = [
    # Batch 0 — Detection & Localization
    ["object_detection", "object_counting", "person_detection",
     "face_detection", "vehicle_detection", "animal_detection",
     "text_detection_ocr", "logo_detection", "anomaly_detection", "visual_grounding"],

    # Batch 1 — Segmentation & Matting
    ["semantic_segmentation", "instance_segmentation", "panoptic_segmentation",
     "background_removal", "salient_object_segmentation", "matting",
     "image_inpainting", "image_outpainting", "change_detection",
     "medical_image_segmentation"],

    # Batch 2 — Generation, Restoration & 3D
    ["text_to_image_generation", "style_transfer", "super_resolution",
     "image_colorization", "image_denoising", "image_deblurring",
     "depth_estimation", "surface_normal_estimation", "3d_reconstruction",
     "point_cloud_generation"],

    # Batch 3 — Classification & Recognition
    ["image_classification", "fine_grained_classification", "scene_recognition",
     "facial_expression_recognition", "age_gender_estimation", "action_recognition",
     "pose_estimation", "gaze_estimation", "attribute_recognition", "stereo_matching"],

    # Batch 4 — Visual Understanding, Video & Specialized
    ["visual_question_answering", "image_captioning", "visual_reasoning",
     "relation_detection", "video_object_tracking", "optical_flow_estimation",
     "video_captioning", "temporal_action_localization",
     "document_layout_analysis", "satellite_image_analysis"],
]

all_batched = [t for batch in TOOL_BATCHES for t in batch]
assert len(all_batched) == 50
assert set(all_batched) == set(TOOLS)


def get_tool_list_str() -> str:
    """Formatted tool list for LLM prompts."""
    return "\n".join(
        f"  {i:02d}. {t} — {TOOL_DESCRIPTIONS[t]}"
        for i, t in enumerate(TOOLS)
    )
