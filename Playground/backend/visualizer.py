import base64
import io
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


# ── Helpers ────────────────────────────────────────────────────────────────────

def _palette(n: int) -> list[tuple[int, int, int]]:
    """Return n visually distinct RGB colors using the golden-angle heuristic."""
    colors: list[tuple[int, int, int]] = []
    for i in range(max(n, 1)):
        hue = (i * 137.508) % 360
        h = hue / 60.0
        c = 200
        x = int(c * (1 - abs(h % 2 - 1)))
        c = int(c)
        sector = int(h)
        if sector == 0:   rgb = (c, x, 0)
        elif sector == 1: rgb = (x, c, 0)
        elif sector == 2: rgb = (0, c, x)
        elif sector == 3: rgb = (0, x, c)
        elif sector == 4: rgb = (x, 0, c)
        else:             rgb = (c, 0, x)
        colors.append(rgb)
    return colors


def _load_font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Drawing functions ──────────────────────────────────────────────────────────

def draw_bboxes(image: Image.Image, bboxes: list, labels: list[str]) -> Image.Image:
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(13)
    colors = _palette(len(bboxes))

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = (int(v) for v in bbox)
        r, g, b = colors[i % len(colors)]

        # Semi-transparent fill + solid border
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 45), outline=(r, g, b, 230), width=2)

        # Label pill
        text = label[:40]
        bbox_text = font.getbbox(text)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        pad = 4
        lx1, ly1 = x1, max(0, y1 - th - pad * 2)
        lx2, ly2 = x1 + tw + pad * 2, y1
        draw.rectangle([lx1, ly1, lx2, ly2], fill=(r, g, b, 210))
        draw.text((lx1 + pad, ly1 + pad - 1), text, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_polygons(image: Image.Image, polygons: list, labels: list[str]) -> Image.Image:
    """polygons: list of polygon groups, each group is a list of flat [x,y,x,y,...] coords."""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = _palette(len(polygons))

    for i, (poly_group, _label) in enumerate(zip(polygons, labels)):
        r, g, b = colors[i % len(colors)]
        for flat in poly_group:
            pts = [(int(flat[j]), int(flat[j + 1])) for j in range(0, len(flat) - 1, 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=(r, g, b, 90), outline=(r, g, b, 240))

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_multi_instance_segmentation(
    image: Image.Image,
    polygons: list,
    bboxes: list,
    labels: list[str],
) -> Image.Image:
    """
    Composite renderer for multi-instance segmentation cascade.
    Per instance: fills the polygon mask, outlines the bbox, and stamps a label pill.
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(13)
    colors = _palette(len(polygons))

    for i, poly_group in enumerate(polygons):
        r, g, b = colors[i % len(colors)]

        # Polygon mask (each instance may have multiple disconnected pieces)
        for flat in poly_group:
            pts = [(int(flat[j]), int(flat[j + 1])) for j in range(0, len(flat) - 1, 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=(r, g, b, 110), outline=(r, g, b, 245))

        # Instance bbox outline + label
        if i < len(bboxes):
            x1, y1, x2, y2 = (int(v) for v in bboxes[i])
            draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 230), width=2)

            label = labels[i] if i < len(labels) else ""
            text = f"#{i + 1} {label}"[:40]
            tb = font.getbbox(text)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad = 4
            lx1, ly1 = x1, max(0, y1 - th - pad * 2)
            lx2, ly2 = x1 + tw + pad * 2, y1
            draw.rectangle([lx1, ly1, lx2, ly2], fill=(r, g, b, 220))
            draw.text((lx1 + pad, ly1 + pad - 1), text, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_ocr_bboxes(image: Image.Image, quad_boxes: list, labels: list[str]) -> Image.Image:
    """quad_boxes: list of [x1,y1,x2,y2,x3,y3,x4,y4] (4-point quads)."""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(11)
    accent = (255, 165, 0)  # orange

    for quad, label in zip(quad_boxes, labels):
        pts = [(int(quad[j]), int(quad[j + 1])) for j in range(0, 8, 2)]
        draw.polygon(pts, outline=(*accent, 230), fill=(*accent, 30))
        x_min = min(p[0] for p in pts)
        y_min = min(p[1] for p in pts) - 14
        draw.text((x_min + 2, max(0, y_min)), label[:50], fill=(*accent, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


# ── Main router ────────────────────────────────────────────────────────────────

def visualize(task_key: str, parsed_output: dict, image: Image.Image) -> Optional[Image.Image]:
    """
    Route to the right drawing function.
    Returns None for text-only tasks (captions, plain OCR, region category/description).
    """
    # Cascaded tasks store grounding under the grounding key
    if task_key in ("<CAPTION_GROUNDING>", "<DETAILED_CAPTION_GROUNDING>", "<MORE_DETAILED_CAPTION_GROUNDING>"):
        data = parsed_output.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        if isinstance(data, dict) and "bboxes" in data:
            return draw_bboxes(image, data["bboxes"], data.get("labels", []))
        return None

    data = parsed_output.get(task_key, {})

    # ── bbox tasks ────────────────────────────────────────────────────────────
    if task_key in (
        "<OD>",
        "<DENSE_REGION_CAPTION>",
        "<REGION_PROPOSAL>",
        "<CAPTION_TO_PHRASE_GROUNDING>",
        "<OPEN_VOCABULARY_DETECTION>",
    ):
        if isinstance(data, dict) and "bboxes" in data:
            labels = data.get("labels", [""] * len(data["bboxes"]))
            return draw_bboxes(image, data["bboxes"], labels)

    # ── segmentation tasks ────────────────────────────────────────────────────
    elif task_key in ("<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"):
        if isinstance(data, dict) and "polygons" in data:
            labels = data.get("labels", [""] * len(data["polygons"]))
            return draw_polygons(image, data["polygons"], labels)

    # ── multi-instance cascade ────────────────────────────────────────────────
    elif task_key == "<MULTI_INSTANCE_SEGMENTATION>":
        if isinstance(data, dict) and data.get("polygons"):
            return draw_multi_instance_segmentation(
                image,
                data["polygons"],
                data.get("instance_bboxes", []),
                data.get("labels", []),
            )

    # ── OCR with region ───────────────────────────────────────────────────────
    elif task_key == "<OCR_WITH_REGION>":
        if isinstance(data, dict) and "quad_boxes" in data:
            return draw_ocr_bboxes(image, data["quad_boxes"], data.get("labels", []))

    # Text-only tasks (<CAPTION>, <OCR>, <REGION_TO_CATEGORY>, etc.) return None
    return None
