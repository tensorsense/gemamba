from transformers import ProcessorMixin
import torch


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VideoMambaProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]
    video_processor_class = "VivitImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, video_processor=None, tokenizer=None, **kwargs):

        if video_processor is None:
            raise ValueError("You need to specify an `video_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(video_processor=video_processor, tokenizer=tokenizer)

    def __call__(self, text=None, videos=None, return_tensors="pt", **kwargs):
        if text is None and videos is None:
            raise ValueError(
                "You have to specify either text or images. Both cannot be none."
            )

        encoding = (
            self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            if text is not None
            else {}
        )

        if videos is not None:
            pixel_values = [
                self.video_processor(
                    list(video),
                    return_tensors=return_tensors,

                )["pixel_values"]
                for video in videos
            ]

            if return_tensors == "pt":
                pixel_values = torch.vstack(pixel_values)
                pixel_values = pixel_values.permute(
                    0, 2, 1, 3, 4
                )  # (B, C, T, H, W), torch.uint8 for 3d conv to work
            else:
                raise NotImplementedError

        return encoding | {"pixel_values": pixel_values}
