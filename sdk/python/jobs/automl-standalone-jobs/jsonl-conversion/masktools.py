import numpy
from pycocotools import mask as pycoco_mask
from simplification.cutil import simplify_coords
from skimage import measure


class MaskToolsLiterals:
    """String keys for Mask tool parameters"""

    MASK_PIXEL_SCORE_THRESHOLD = "mask_pixel_score_threshold"
    MAX_NUMBER_OF_POLYGON_POINTS = "max_number_of_polygon_points"


class MaskToolsParameters:
    """Default values for mask tool parameters."""

    DEFAULT_MASK_PIXEL_SCORE_THRESHOLD = 0.5
    DEFAULT_MAX_NUMBER_OF_POLYGON_POINTS = 100
    DEFAULT_MAX_NUMBER_OF_POLYGON_SIMPLIFICATIONS = 25
    DEFAULT_MASK_SAFETY_PADDING = 1
    DEFAULT_GRABCUT_MARGIN = 10
    DEFAULT_GRABCUT_MODEL_LEVELS = 65
    DEFAULT_GRABCUT_NUMBER_ITERATIONS = 5
    DEFAULT_MASK_REFINE_POINTS = 25


def convert_mask_to_polygon(
    rle_mask,
    max_polygon_points=MaskToolsParameters.DEFAULT_MAX_NUMBER_OF_POLYGON_POINTS,
    max_refinement_iterations=MaskToolsParameters.DEFAULT_MAX_NUMBER_OF_POLYGON_SIMPLIFICATIONS,
    edge_safety_padding=MaskToolsParameters.DEFAULT_MASK_SAFETY_PADDING,
):
    """Convert a run length encoded mask to a polygon outline in normalized coordinates.

    :param rle_mask: Run length encoding of a binary mask
    :type: rle_mask: <class 'dict'>
    :param max_polygon_points: Maximum number of (x, y) coordinate pairs in polygon
    :type: max_polygon_points: int
    :param max_refinement_iterations: Maximum number of times to refine the polygon
    trying to reduce the number of pixels to meet max polygon points.
    :type: max_refinement_iterations: int
    :param edge_safety_padding: Number of pixels to pad the mask with
    :type edge_safety_padding: int
    :return: normalized polygon coordinates
    :rtype: list of list
    """
    # Convert rle mask to numpy bitmask
    mask_array = decode_rle_masks_as_binary_mask([rle_mask])
    image_shape = mask_array.shape

    # Pad the mask to avoid errors at the edge of the mask
    embedded_mask = numpy.zeros(
        (
            image_shape[0] + 2 * edge_safety_padding,
            image_shape[1] + 2 * edge_safety_padding,
        ),
        dtype=numpy.uint8,
    )
    embedded_mask[
        edge_safety_padding : image_shape[0] + edge_safety_padding,
        edge_safety_padding : image_shape[1] + edge_safety_padding,
    ] = mask_array

    # Find Image Contours
    contours = measure.find_contours(embedded_mask, 0.5)
    simplified_contours = []

    for contour in contours:

        # Iteratively reduce polygon points, if necessary
        if max_polygon_points is not None:
            simplify_factor = 0
            while (
                len(contour) > max_polygon_points
                and simplify_factor < max_refinement_iterations
            ):
                contour = simplify_coords(contour, simplify_factor)
                simplify_factor += 1

        # Convert to [x, y, x, y, ....] coordinates and correct for padding
        unwrapped_contour = [0] * (2 * len(contour))
        unwrapped_contour[::2] = numpy.ceil(contour[:, 1]) - edge_safety_padding
        unwrapped_contour[1::2] = numpy.ceil(contour[:, 0]) - edge_safety_padding

        simplified_contours.append(unwrapped_contour)

    return _normalize_contour(simplified_contours, image_shape)


def decode_rle_masks_as_binary_mask(rle_masks):
    """Decode list of run-length encoded masks representing a single outline to a binary mask.

    :param rle_masks: List of run-length encoded masks
    :type rle_masks: List <class 'dict'>
    :return: Segmentation mask
    :rtype: height x width numpy array
    """
    if not rle_masks:
        return None

    masks = []

    for rle in rle_masks:
        m = pycoco_mask.decode(rle)
        masks.append(m)

    # Overlapping segments are used to represent holes in the
    # mask - exclusive or cuts these holes in the bit masks
    base_mask = masks[0]

    for m in masks[1:]:
        base_mask = numpy.logical_xor(base_mask, m)

    return base_mask


def _normalize_contour(contours, image_shape):

    height, width = image_shape[0], image_shape[1]

    for contour in contours:
        contour[::2] = [x * 1.0 / width for x in contour[::2]]
        contour[1::2] = [y * 1.0 / height for y in contour[1::2]]

    return contours


def encode_mask_as_rle(mask):
    """Encode binary mask via run-length encoding.

    :param mask: Binary mask in torch.Size([height, width]) or torch.Size([1, height, width])
    :type: mask: <class 'torch.Tensor'>
    :return: Run length encoding of the binary mask
    :rtype: <class 'dict'>
    """
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)

    rle = pycoco_mask.encode(
        numpy.array(mask[0, :, :, numpy.newaxis], dtype=numpy.uint8, order="F")
    )[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
