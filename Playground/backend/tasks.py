from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskGroup(str, Enum):
    CAPTION = "caption"
    DETECTION = "detection"
    GROUNDING = "grounding"
    SEGMENTATION = "segmentation"
    OCR = "ocr"
    CASCADED = "cascaded"


@dataclass
class TaskInfo:
    key: str
    name: str
    group: TaskGroup
    description: str
    example: str
    needs_text_input: bool = False
    needs_region_input: bool = False
    text_input_placeholder: Optional[str] = None


TASKS: list[TaskInfo] = [
    # ── Caption ────────────────────────────────────────────────────────────────
    TaskInfo(
        key="<CAPTION>",
        name="Caption",
        group=TaskGroup.CAPTION,
        description="Generate a short, concise one-line description of the image.",
        example="A dog sitting on a park bench.",
    ),
    TaskInfo(
        key="<DETAILED_CAPTION>",
        name="Detailed Caption",
        group=TaskGroup.CAPTION,
        description="Generate a detailed paragraph-level description of the image.",
        example="A golden retriever is sitting on a wooden park bench surrounded by autumn leaves.",
    ),
    TaskInfo(
        key="<MORE_DETAILED_CAPTION>",
        name="More Detailed Caption",
        group=TaskGroup.CAPTION,
        description="Generate an exhaustive, richly detailed description covering every visual element.",
        example="In this outdoor autumn scene, a golden retriever with a shiny golden coat...",
    ),
    # ── Detection ──────────────────────────────────────────────────────────────
    TaskInfo(
        key="<OD>",
        name="Object Detection",
        group=TaskGroup.DETECTION,
        description="Detect all objects in the image and draw labelled bounding boxes around them.",
        example="Detects: person, car, bicycle, traffic light...",
    ),
    TaskInfo(
        key="<DENSE_REGION_CAPTION>",
        name="Dense Region Caption",
        group=TaskGroup.DETECTION,
        description="Generate a descriptive caption for every detected region in the image.",
        example="Region 1: a golden retriever, Region 2: a wooden park bench...",
    ),
    TaskInfo(
        key="<REGION_PROPOSAL>",
        name="Region Proposal",
        group=TaskGroup.DETECTION,
        description="Propose candidate bounding box regions of interest without class labels.",
        example="Returns salient regions as unlabelled bounding boxes.",
    ),
    TaskInfo(
        key="<OPEN_VOCABULARY_DETECTION>",
        name="Open Vocabulary Detection",
        group=TaskGroup.DETECTION,
        description="Detect specific objects you name in the text prompt — not limited to preset classes.",
        example="Type 'a dog and a frisbee' to locate exactly those objects.",
        needs_text_input=True,
        text_input_placeholder="e.g. a dog and a frisbee",
    ),
    # ── Grounding ──────────────────────────────────────────────────────────────
    TaskInfo(
        key="<CAPTION_TO_PHRASE_GROUNDING>",
        name="Caption → Phrase Grounding",
        group=TaskGroup.GROUNDING,
        description="Provide a caption and the model localizes each noun phrase to its region in the image.",
        example="Input: 'A dog on a bench' → draws boxes around 'dog' and 'bench'.",
        needs_text_input=True,
        text_input_placeholder="e.g. A dog sitting on a bench in a park",
    ),
    TaskInfo(
        key="<REGION_TO_CATEGORY>",
        name="Region → Category",
        group=TaskGroup.GROUNDING,
        description="Classify the object inside a bounding box region you specify.",
        example="Provide pixel coordinates of a box to get the object's category label.",
        needs_region_input=True,
        text_input_placeholder="x1,y1,x2,y2  (pixel coordinates)",
    ),
    TaskInfo(
        key="<REGION_TO_DESCRIPTION>",
        name="Region → Description",
        group=TaskGroup.GROUNDING,
        description="Get a natural-language description of whatever is inside a bounding box you draw.",
        example="Provide pixel coordinates to describe the content of that region.",
        needs_region_input=True,
        text_input_placeholder="x1,y1,x2,y2  (pixel coordinates)",
    ),
    # ── Segmentation ───────────────────────────────────────────────────────────
    TaskInfo(
        key="<REFERRING_EXPRESSION_SEGMENTATION>",
        name="Referring Expression Segmentation",
        group=TaskGroup.SEGMENTATION,
        description="Segment the specific object described by your text expression.",
        example="Type 'the dog on the left' to get a pixel mask for that dog only.",
        needs_text_input=True,
        text_input_placeholder="e.g. the red car on the left",
    ),
    TaskInfo(
        key="<REGION_TO_SEGMENTATION>",
        name="Region → Segmentation",
        group=TaskGroup.SEGMENTATION,
        description="Produce a precise segmentation mask for the object inside a bounding box you specify.",
        example="Provide pixel coordinates of a box to get its precise polygon mask.",
        needs_region_input=True,
        text_input_placeholder="x1,y1,x2,y2  (pixel coordinates)",
    ),
    TaskInfo(
        key="<MULTI_INSTANCE_SEGMENTATION>",
        name="Multi-Instance Segmentation",
        group=TaskGroup.SEGMENTATION,
        description="Segment EVERY instance of an object class. Two-stage cascade: phrase-grounding finds all instances, then per-region segmentation produces a mask for each.",
        example="Type 'plastic bag' to get a polygon mask for every plastic bag in the image.",
        needs_text_input=True,
        text_input_placeholder="e.g. plastic bag, drum, car",
    ),
    # ── OCR ────────────────────────────────────────────────────────────────────
    TaskInfo(
        key="<OCR>",
        name="OCR",
        group=TaskGroup.OCR,
        description="Extract all text visible in the image as a plain string.",
        example="Extracts text from signs, documents, labels, whiteboards, etc.",
    ),
    TaskInfo(
        key="<OCR_WITH_REGION>",
        name="OCR with Region",
        group=TaskGroup.OCR,
        description="Extract text and return the bounding box location of each text block.",
        example="Returns each word/line with its exact position drawn on the image.",
    ),
    # ── Cascaded ───────────────────────────────────────────────────────────────
    TaskInfo(
        key="<CAPTION_GROUNDING>",
        name="Caption + Grounding",
        group=TaskGroup.CASCADED,
        description="Two-step pipeline: auto-caption the image, then ground every phrase to its region.",
        example="Step 1: generate caption → Step 2: localize each phrase as a bounding box.",
    ),
    TaskInfo(
        key="<DETAILED_CAPTION_GROUNDING>",
        name="Detailed Caption + Grounding",
        group=TaskGroup.CASCADED,
        description="Two-step pipeline using the detailed caption, then grounding all phrases.",
        example="More descriptive output with full phrase localization overlaid on the image.",
    ),
    TaskInfo(
        key="<MORE_DETAILED_CAPTION_GROUNDING>",
        name="More Detailed Caption + Grounding",
        group=TaskGroup.CASCADED,
        description="Two-step pipeline using the richest caption, then grounding every phrase.",
        example="Maximum detail captioning with complete phrase-to-region visualization.",
    ),
]

TASK_MAP: dict[str, TaskInfo] = {t.key: t for t in TASKS}
TASK_GROUPS: dict[TaskGroup, list[TaskInfo]] = {
    g: [t for t in TASKS if t.group == g] for g in TaskGroup
}

# Cascaded task → first-step caption task mapping
CASCADED_MAP: dict[str, str] = {
    "<CAPTION_GROUNDING>": "<CAPTION>",
    "<DETAILED_CAPTION_GROUNDING>": "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION_GROUNDING>": "<MORE_DETAILED_CAPTION>",
}
