from PIL import Image
from typing import Dict, Any, Tuple, Optional, List, Union
import re

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# bbox settings
PHRASE_ST_PLACEHOLDER = '<ph_st>'
PHRASE_ED_PLACEHOLDER = '<ph_ed>'
IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
OBJS_PLACEHOLDER = '<objs>'
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'

roi_start_tag='<roi>'
roi_end_tag='</roi>'
roi_pad_tag='<roipad>'

box_start_tag='<box>'
box_end_tag='</box>'
box_pad_tag='<boxpad>'

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box

def point_xy_expand2square(point, *, w, h):
    pseudo_box = (point[0], point[1], point[0], point[1])
    expanded_box = box_xyxy_expand2square(box=pseudo_box, w=w, h=h)
    expanded_point = (expanded_box[0], expanded_box[1])
    return expanded_point

class PreProcess_boxplaceholder:
    def __init__(self, roi_token_len=1, box_token_len=5, use_roi_startend=False, use_box_startend=True):
        self.box_token_len = box_token_len
        self.roi_token_len = roi_token_len
        self.use_roi_startend = use_roi_startend
        self.use_box_startend = use_box_startend

    def process_question_roi(self, sentence):
        replace_token = roi_pad_tag * self.roi_token_len
        if self.use_roi_startend:
            replace_token = roi_start_tag + replace_token + roi_end_tag
        sentence = sentence.replace(BOXES_PLACEHOLDER, replace_token)
        return sentence
    
    def process_answer_query(self, sentence):
        replace_token = box_pad_tag * self.box_token_len
        if self.use_box_startend:
            replace_token = box_start_tag + replace_token + box_end_tag
        sentence = sentence.replace(BOXES_PLACEHOLDER, replace_token)
        return sentence
    
             
    def __call__(self, sentence, type="question"):
        if type in ["question"]:
            return self.process_question_roi(sentence)
        elif type in ["answer"]:
            return self.process_answer_query(sentence)
           


class Expand2square:
    def __init__(self, background_color=(255, 255, 255)):
        self.background_color = background_color

    def __call__(self, image: Image.Image, labels: Dict[str, Any] = None) -> Tuple[
        Image.Image, Optional[Dict[str, Any]]]:
        width, height = image.size
        processed_image = expand2square(image, background_color=self.background_color)
        if labels is None:
            return processed_image, labels
        if 'boxes' in labels:
            bboxes = [box_xyxy_expand2square(bbox, w=width, h=height) for bbox in labels['boxes']]
            labels['boxes'] = bboxes
        if 'points' in labels:
           
            if labels['points']:
                points = [point_xy_expand2square(point, w=width, h=height) for point in labels['points']]
                labels['points'] = points
        return processed_image, labels

class BoxSeq():

    def norm_box_xyxy(self, box, *, w, h):
        x1, y1, x2, y2 = box

        # Calculate the normalized coordinates with min-max clamping
        norm_x1 = max(0.0, min(x1 / w, 1.0))
        norm_y1 = max(0.0, min(y1 / h, 1.0))
        norm_x2 = max(0.0, min(x2 / w, 1.0))
        norm_y2 = max(0.0, min(y2 / h, 1.0))

        # Return the normalized box coordinates
        normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
        return normalized_box
    def map_obj(self, boxes_value: List[List[float]], boxes_seq: List[List[int]]) -> List[List[List[float]]]:
        """
        >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
        >>> boxes_seq_ = [[3, 1], [2]]
        >>> var = map_obj(normalized_boxes, boxes_seq_)
        >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
        """
        try:
            ret = []
            for boxes in boxes_seq:
                boxes_ret = []
                for box_index in boxes:
                    if isinstance(box_index, (list, tuple)):
                        boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                    else:
                        boxes_ret.append(boxes_value[box_index])
                ret.append(boxes_ret)
            return ret
        except:
            raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")

    def __call__(self, boxes_seq, target):
        # box_formatter = preprocessor['target']['boxes']


        # convert bboxes_seq
        normalized_boxes = []
        if target is not None and 'boxes' in target:
            for box in target['boxes']:
                normalized_boxes.append(
                    self.norm_box_xyxy(box, w=target['width'], h=target['height'])
                )
        
        if boxes_seq is not None:
            # map box seq
            # print('ori boxes_seq:',boxes_seq)
            boxes_seq = self.map_obj(normalized_boxes, boxes_seq)
            # reformat; replace <boxes> placeholder
            # print('boxes_seq:',boxes_seq)
            return boxes_seq



class BoxFormatProcess():
    def __init__(self, box_formatter):
        self.box_formatter = box_formatter

    def map_obj(self, boxes_value: List[List[float]], boxes_seq: List[List[int]]) -> List[List[List[float]]]:
        """
        >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
        >>> boxes_seq_ = [[3, 1], [2]]
        >>> var = map_obj(normalized_boxes, boxes_seq_)
        >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
        """
        try:
            ret = []
            for boxes in boxes_seq:
                boxes_ret = []
                for box_index in boxes:
                    if isinstance(box_index, (list, tuple)):
                        boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                    else:
                        boxes_ret.append(boxes_value[box_index])
                ret.append(boxes_ret)
            return ret
        except:
            raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")

    def norm_box_xyxy(self, box, *, w, h):
        x1, y1, x2, y2 = box

        # Calculate the normalized coordinates with min-max clamping
        norm_x1 = max(0.0, min(x1 / w, 1.0))
        norm_y1 = max(0.0, min(y1 / h, 1.0))
        norm_x2 = max(0.0, min(x2 / w, 1.0))
        norm_y2 = max(0.0, min(y2 / h, 1.0))

        # Return the normalized box coordinates
        normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
        return normalized_box

    def norm_point_xyxy(self, point, *, w, h):
        x, y = point
        norm_x = max(0.0, min(x / w, 1.0))
        norm_y = max(0.0, min(y / h, 1.0))
        point = norm_x, norm_y
        return point

    def __call__(self, sentence: Dict[str, Any], target: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # box_formatter = preprocessor['target']['boxes']

        normalized_boxes = []
        # print('target:', target['boxes'])
        if target is not None and 'boxes' in target:
            for box in target['boxes']:
                normalized_boxes.append(
                    self.norm_box_xyxy(box, w=target['width'], h=target['height'])
                )
        normalized_points = []
        if target is not None and 'points' in target:
            for point in target['points']:
                normalized_points.append(
                    self.norm_point_xyxy(point, w=target['width'], h=target['height'])
                )

        # convert bboxes_seq
        words: str = sentence['value']
        boxes_seq: List[List[int]] = sentence.get('boxes_seq', None)
        if boxes_seq is not None:
            # map box seq
            boxes_seq = self.map_obj(normalized_boxes, boxes_seq)
            # reformat; replace <boxes> placeholder
            converted = self.box_formatter(words, boxes_seq)
            
            words = converted
        points_seq: List[List[int]] = sentence.get('points_seq', None)
        if points_seq is not None:
            # map point seq
            points_seq: List[Boxes] = self.map_obj(normalized_points, points_seq)
            # reformat; replace <points> placeholder
            converted = self.box_formatter.call_on_point(words, points_seq)
            words = converted

        if boxes_seq is not None or points_seq is not None:
            sentence['raw_value'] = sentence['value']
            sentence['value'] = words
        return sentence, target


class BoxFormatter:
    def __init__(self, bboxes_token=BOXES_PLACEHOLDER, points_token=POINTS_PLACEHOLDER):
        self.bboxes_token = bboxes_token
        self.points_token = points_token
        # normally the bboxes_token_pat is the same as bboxes_token if u not use some weird token
        self.bboxes_token_pat = re.compile(bboxes_token)
        self.points_token_pat = re.compile(points_token)

    def __call__(self, sentence: str, bboxes_seq: BoxesSeq) -> str:
        all_box = self.bboxes_token_pat.findall(sentence)
        assert len(all_box) == len(bboxes_seq), f"not match. sentence: {sentence}. boxes:{bboxes_seq}, all_box:{all_box}"
        # print('bboxes_seq: ',bboxes_seq)
        # print('all_box: ',all_box)
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_box(bboxes) for bboxes in bboxes_seq]
        converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
        return converted

    def call_on_point(self, sentence: str, points_seq: BoxesSeq) -> str:
        all_box = self.points_token_pat.findall(sentence)
        assert len(all_box) == len(points_seq), f"not match. sentence: {sentence}. boxes:{points_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_point(bboxes) for bboxes in points_seq]
        converted = sentence.replace(self.points_token, '{}').format(*bboxes_strs)
        return converted

    def format_point(self, points) -> str:
        raise NotImplementedError

    def format_box(self, bboxes: Boxes) -> str:
        raise NotImplementedError

    def extract(self, string: str) -> List[Boxes]:
        raise NotImplementedError

    def extract_point(self, string: str) -> List[Boxes]:
        raise NotImplementedError


class PlainBoxFormatter(BoxFormatter):

    def __init__(self, *args, precision=3, use_small_brackets=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.use_small_brackets = use_small_brackets

        small_brackets_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\)')
        small_brackets_point_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\)')

        middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')
        middle_brackets_point_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\]')

        self.pat = small_brackets_pat if use_small_brackets else middle_brackets_pat
        self.point_pat = small_brackets_point_pat if use_small_brackets else middle_brackets_point_pat

    def format_box(self, boxes: Boxes) -> str:
        box_strs = []
        for box in boxes:
            box_strs.append(','.join([f"{elem:.{self.precision}f}" for elem in box]))
        box_str = ';'.join(box_strs)
        if self.use_small_brackets:
            return "(" + box_str + ")"
        return "[" + box_str + "]"

    def format_point(self, points) -> str:
        return self.format_box(points)

    def extract(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    def extract_point(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.point_pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret