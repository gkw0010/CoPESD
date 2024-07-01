from typing import Tuple
from PIL import Image
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def select_best_size_match(pil_image, crop_size_list):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    best_sizes = [(x, y) for x, y in zip(rem_percent, crop_size_list)]
    best_size = max(best_sizes)
    best_size= best_size[1]
    return best_size


def generate_candidate_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.
    crop_size_list = []
    wp, hp = num_patches, 1 #(9,1)
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list

class PadToSquare:
    def __init__(self, background_color:Tuple[float, float, float]):
        """
        pad an image to squre (borrowed from LLAVA, thx)
        :param background_color: rgb values for padded pixels, normalized to [0, 1]
        """
        self.bg_color = tuple(int(x*255) for x in background_color)

    def __call__(self, img: Image.Image):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.bg_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.bg_color)
            result.paste(img, ((height - width) // 2, 0))
            return result

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(bg_color={self.bg_color})"
        return format_string


def T_random_resized_crop(size=224):
    t = transforms.Compose([
        transforms.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                     antialias=None),  # 3 is bicubic
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t


def T_resized_center_crop(size=224):
    t = transforms.Compose([
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t


def T_padded_resize(size=224):
    t = transforms.Compose([
        PadToSquare(background_color=(0.48145466, 0.4578275, 0.40821073)),
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t

def only_norm(size=224):
    t = transforms.Compose([
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t

class AnyResTransform:
    def __init__(self, grid_size=224, max_views=9, max_ratio=4):
        self.grid_size = grid_size
        self.max_views = max_views
        self.max_ratio = max_ratio

        self._image_transform = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.candidate_sizes = generate_candidate_size_list(max_views, grid_size, max_ratio)

    def _process(self, image: Image.Image):
        w, h = image.size
        assert w > 0
        assert h > 0

        target_size = select_best_size_match(image, self.candidate_sizes)
        
        image_scale = min(target_size[0] / w, target_size[1] / h)
        rescaled_image = image.resize(
            (round(w * image_scale), round(h * image_scale)),
            Image.Resampling.BICUBIC
        )
        target_image = Image.new('RGB', target_size)
        target_image.paste(rescaled_image, (0, 0))

        assert int(target_size[0]) % self.grid_size == 0 and int(target_size[1]) % self.grid_size == 0
        w_grids = target_size[0] // self.grid_size
        h_grids = target_size[1] // self.grid_size
        num_views = w_grids * h_grids

        # local views
        l_local = []
        l_local_tensor = []
        if num_views > 1:
            for i in range(h_grids):
                for j in range(w_grids):
                    box = (j*self.grid_size, i*self.grid_size, (j+1)*self.grid_size, (i+1)*self.grid_size)
                    local_image = target_image.crop(box)
                    l_local.append(local_image)
                    tmp = self._image_transform(local_image)
                    # print('tmp', tmp.shape)
                    l_local_tensor.append(tmp)

        # square global view
        global_scale = image_scale / max(w_grids, h_grids)
        global_rescaled_image = image.resize(
            (round(w * global_scale), round(h * global_scale)),
            Image.Resampling.BICUBIC
        )
        global_square_image = Image.new('RGB', (self.grid_size, self.grid_size))
        global_square_image.paste(global_rescaled_image, (0, 0))
        global_square_tensor = self._image_transform(global_square_image)
        # print('global', global_square_tensor.shape)

        # todo deal_with_boxes
        return {
            "w_grids": w_grids,
            "h_grids": h_grids,
            # "global": global_square_image,
            "global_tensor": global_square_tensor,
            # "scale_to_global_view": global_scale,
            # "l_local": l_local,
            "l_local_tensor": l_local_tensor,
        }

    def __call__(self, image: Image.Image):
        # todo make a class called image with boxes and points
        # image = image.convert('RGB')
        item = self._process(image)
        return item


def get_transform(transform_type: str, size=224, max_views=9):
    if transform_type == "random_resized_crop":
        transform = T_random_resized_crop(size)
    elif transform_type == "resized_center_crop":
        transform = T_resized_center_crop(size)
    elif transform_type == "padded_resize":
        transform = T_padded_resize(size)
    elif transform_type == "only_norm":
        transform = only_norm(size)        
    elif transform_type == "anyres":
        transform = AnyResTransform(grid_size=224, max_views=max_views, max_ratio=4)       
    else:
        raise ValueError("unknown transform type: transform_type")
    return transform