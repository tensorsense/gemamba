from transformers import ProcessorMixin
from torchvision import transforms
import decord
from decord import VideoReader
import torch
import numpy as np
import random

decord.bridge.set_bridge("torch")

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_video_transform(config):
    normalize = transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    transform = transforms.Compose(
        [
            transforms.Resize(
                (config.inputs.image_res, config.inputs.image_res),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )
    return transform


def get_frame_indices(
    num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1
):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = (
            1 / output_fps
        )  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_decord(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    max_num_frames=-1,
    client=None,
    trimmed30=False,
    transform=None,
):
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # only use top 30 seconds
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames,
        vlen,
        sample=sample,
        fix_start=fix_start,
        input_fps=fps,
        max_num_frames=max_num_frames,
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    if transform is not None:
        frames = transform(frames)
    return frames  # frame_indices, duration


class VideoMambaVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = "VideoMambaVideoProcessor"

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_video_transform(config)
        # self.image_processor = load_and_transform_video
        self.video_reader = read_frames_decord
        self.tokenizer = tokenizer

    def __call__(
        self, images=None, text=None, context_length=77, return_tensors=None, **kwargs
    ):
        if text is None and images is None:
            raise ValueError(
                "You have to specify either text or images. Both cannot be none."
            )

        if text is not None:
            encoding = self.tokenizer(
                text,
                max_length=context_length,
                padding="max_length",
                truncation=True,
                return_tensors=return_tensors,
                **kwargs
            )

        if images is not None:
            if not isinstance(images, list):
                images = [images]

            image_features = [
                self.video_reader(
                    image,
                    config.inputs.video_input.num_frames,
                    config.inputs.video_input.sample_type,
                    max_num_frames=-1,
                    client=None,
                    trimmed30=False,
                    transform=self.transform,
                )
                for image in images
            ]
            # image_features = [
            #     self.image_processor(
            #         image,
            #         self.transform,
            #         video_decode_backend=self.config.vision_config.video_decode_backend,
            #         num_frames=self.config.vision_config.num_frames,
            #     )
            #     for image in images
            # ]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(
            *args, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(
            *args, skip_special_tokens=skip_special_tokens, **kwargs
        )
