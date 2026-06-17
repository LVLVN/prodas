import cv2
import numpy as np

FABRIC_CATEGORY_MAPPING = {
    '线头': 'loose thread',
    '断经': 'warp breakage',
    '短纬': 'short weft',
    '并纬': 'double weft',
    '线头断经': 'thread and warp breakage'
}

DEFECT_TEMPLATES = {
    'loose thread': {
        'center-middle': 'loose thread defect in the center of the fabric',
        'left-middle': 'loose thread defect on the left side',
        'right-middle': 'loose thread defect on the right side',
        'center-top': 'loose thread defect at the top center',
        'center-bottom': 'loose thread defect at the bottom center',
        'left-top': 'loose thread defect in the top left corner',
        'right-top': 'loose thread defect in the top right corner',
        'left-bottom': 'loose thread defect in the bottom left corner',
        'right-bottom': 'loose thread defect in the bottom right corner'
    },
    'warp breakage': {
        'center-middle': 'vertical warp breakage in the center',
        'left-middle': 'vertical warp breakage on the left side',
        'right-middle': 'vertical warp breakage on the right side',
        'center-top': 'vertical warp breakage starting from the top',
        'center-bottom': 'vertical warp breakage starting from the bottom',
        'left-top': 'vertical warp breakage in the top left area',
        'right-top': 'vertical warp breakage in the top right area',
        'left-bottom': 'vertical warp breakage in the bottom left area',
        'right-bottom': 'vertical warp breakage in the bottom right area'
    },
    'short weft': {
        'center-middle': 'horizontal short weft defect in the center',
        'left-middle': 'horizontal short weft defect on the left side',
        'right-middle': 'horizontal short weft defect on the right side',
        'center-top': 'horizontal short weft defect near the top',
        'center-bottom': 'horizontal short weft defect near the bottom',
        'left-top': 'horizontal short weft defect in the top left area',
        'right-top': 'horizontal short weft defect in the top right area',
        'left-bottom': 'horizontal short weft defect in the bottom left area',
        'right-bottom': 'horizontal short weft defect in the bottom right area'
    },
    'double weft': {
        'center-middle': 'parallel double weft lines in the center',
        'left-middle': 'parallel double weft lines on the left side',
        'right-middle': 'parallel double weft lines on the right side',
        'center-top': 'parallel double weft lines near the top',
        'center-bottom': 'parallel double weft lines near the bottom',
        'left-top': 'parallel double weft lines in the top left area',
        'right-top': 'parallel double weft lines in the top right area',
        'left-bottom': 'parallel double weft lines in the bottom left area',
        'right-bottom': 'parallel double weft lines in the bottom right area'
    },
    'thread and warp breakage': {
        'center-middle': 'combined thread and warp breakage defect in the center',
        'left-middle': 'combined thread and warp breakage defect on the left side',
        'right-middle': 'combined thread and warp breakage defect on the right side',
        'center-top': 'combined thread and warp breakage defect near the top',
        'center-bottom': 'combined thread and warp breakage defect near the bottom',
        'left-top': 'combined thread and warp breakage defect in the top left area',
        'right-top': 'combined thread and warp breakage defect in the top right area',
        'left-bottom': 'combined thread and warp breakage defect in the bottom left area',
        'right-bottom': 'combined thread and warp breakage defect in the bottom right area'
    },
    'crack': {
        'center-middle': 'crack defect in the center of the pavement',
        'left-middle': 'crack defect on the left side of the pavement',
        'right-middle': 'crack defect on the right side of the pavement',
        'center-top': 'crack defect at the top center of the pavement',
        'center-bottom': 'crack defect at the bottom center of the pavement',
        'left-top': 'crack defect in the top left corner of the pavement',
        'right-top': 'crack defect in the top right corner of the pavement',
        'left-bottom': 'crack defect in the bottom left corner of the pavement',
        'right-bottom': 'crack defect in the bottom right corner of the pavement'
    },
    'PCB defect': {
        'center-middle': 'defect in the center of the PCB board',
        'left-middle': 'defect on the left side of the PCB board',
        'right-middle': 'defect on the right side of the PCB board',
        'center-top': 'defect at the top center of the PCB board',
        'center-bottom': 'defect at the bottom center of the PCB board',
        'left-top': 'defect in the top left corner of the PCB board',
        'right-top': 'defect in the top right corner of the PCB board',
        'left-bottom': 'defect in the bottom left corner of the PCB board',
        'right-bottom': 'defect in the bottom right corner of the PCB board'
    },
    'defect': {
        'center-middle': 'surface defect in the center of the commutator',
        'left-middle': 'surface defect on the left side of the commutator',
        'right-middle': 'surface defect on the right side of the commutator',
        'center-top': 'surface defect at the top center of the commutator',
        'center-bottom': 'surface defect at the bottom center of the commutator',
        'left-top': 'surface defect in the top left area of the commutator',
        'right-top': 'surface defect in the top right area of the commutator',
        'left-bottom': 'surface defect in the bottom left area of the commutator',
        'right-bottom': 'surface defect in the bottom right area of the commutator'
    }
}

DEFECT_FALLBACK = {
    'loose thread': 'Unable to detect the precise location of the loose thread',
    'warp breakage': 'Unable to detect the precise location of the warp breakage',
    'short weft': 'Unable to detect the precise location of the short weft',
    'double weft': 'Unable to detect the precise location of the double weft',
    'thread and warp breakage': 'Unable to detect the precise location of the thread and warp breakage',
    'crack': 'Unable to detect the precise location of the crack on the pavement',
    'PCB defect': 'Unable to detect the precise location of the defect on the PCB board',
    'defect': 'A surface defect on the commutator'
}

DEFECT_EXCEPTION_FALLBACK = {
    'crack': 'A crack defect on the pavement surface. This defect appears as a linear pattern on the pavement.',
    'PCB defect': 'A defect on the PCB board surface',
    'defect': 'A surface defect on the commutator'
}


def detect_line_position(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    edges = cv2.Canny(blurred, 100, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=50, maxLineGap=10)

    height, width = image.shape[:2]

    if lines is not None:
        all_centers = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            all_centers.append((center_x, center_y))

        if all_centers:
            avg_center_x = sum(x for x, _ in all_centers) / len(all_centers)
            avg_center_y = sum(y for _, y in all_centers) / len(all_centers)

            position = {
                'x': 'left' if avg_center_x < width/3 else 'right' if avg_center_x > 2*width/3 else 'center',
                'y': 'top' if avg_center_y < height/3 else 'bottom' if avg_center_y > 2*height/3 else 'middle'
            }
            return position, edges, lines

    return None, edges, None


def detect_object_position(image, mask):
    binary = (mask > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(binary,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(max_contour)

        center_x = x + w/2
        center_y = y + h/2

        height, width = image.shape[:2]

        position = {
            'x': 'left' if center_x < width/3 else 'right' if center_x > 2*width/3 else 'center',
            'y': 'top' if center_y < height/3 else 'bottom' if center_y > 2*height/3 else 'middle'
        }

        return position, contours

    return None, None


def normalize_defect_type(defect_type):
    if defect_type in FABRIC_CATEGORY_MAPPING:
        return FABRIC_CATEGORY_MAPPING[defect_type]
    return defect_type


def generate_position_description(position_dict, defect_type):
    defect_type = normalize_defect_type(defect_type)
    templates = DEFECT_TEMPLATES.get(defect_type, DEFECT_TEMPLATES['loose thread'])
    fallback = DEFECT_FALLBACK.get(defect_type, f"Unable to detect the precise location of the {defect_type}")

    if position_dict is None:
        return fallback

    position_key = f"{position_dict['x']}-{position_dict['y']}"

    if defect_type == 'crack':
        default_msg = f"crack defect in {position_dict['x']} {position_dict['y']} area of the pavement"
    elif defect_type == 'PCB defect':
        default_msg = f"defect in {position_dict['x']} {position_dict['y']} area of the PCB board"
    elif defect_type == 'defect':
        default_msg = f"surface defect in {position_dict['x']} {position_dict['y']} area of the commutator"
    else:
        default_msg = f"{defect_type} defect in {position_dict['x']} {position_dict['y']} area"

    return templates.get(position_key, default_msg)


def detect_defect_position(image, mask=None, prefer_line=False, object_only=False):
    if object_only:
        if mask is None:
            return None
        position, _ = detect_object_position(image, mask)
        return position

    if prefer_line:
        position, _, _ = detect_line_position(image)
        if position is not None:
            return position
        if mask is not None:
            position, _ = detect_object_position(image, mask)
            return position
        return None

    if mask is not None:
        position, _ = detect_object_position(image, mask)
        if position is not None:
            return position

    position, _, _ = detect_line_position(image)
    return position


def generate_defect_description(image, mask=None, defect_type='defect', prefer_line=False, object_only=False):
    defect_type = normalize_defect_type(defect_type)
    exception_fallback = DEFECT_EXCEPTION_FALLBACK.get(defect_type)

    try:
        position = detect_defect_position(
            image,
            mask=mask,
            prefer_line=prefer_line,
            object_only=object_only
        )
        description = generate_position_description(position, defect_type)
        if description.startswith("Unable") and exception_fallback:
            return exception_fallback
        return description
    except Exception:
        if exception_fallback:
            return exception_fallback
        return generate_position_description(None, defect_type)
