import imageio
import torch
from torch.autograd import Variable
from tqdm import tqdm
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA, putText, rectangle
from SSD.data import VOC_CLASSES as labelmap
from SSD.data import BaseTransform
from SSD.ssd import build_ssd as SSD


def detect(net, frame, transform):
    height, width = frame.shape[:2]

    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1).unsqueeze(0)
    y = net(x).data

    for i in range(y.size(1)):
        j = 0

        while y[0, i, j, 0] >= 0.6:
            pt = (y[0, i, j, 1:] * torch.Tensor([width, height, width, height])).numpy()

            rectangle(
                frame,
                (int(pt[0]), int(pt[1])),
                (int(pt[2]), int(pt[3])),
                (255, 0, 0),
                2,
            )

            putText(
                frame,
                labelmap[i - 1],
                (int(pt[0]), int(pt[1])),
                FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
                LINE_AA,
            )

            j += 1

    return frame


net = SSD("test")
net.load_state_dict(
    torch.load("Weights.pth", map_location=lambda storage, loc: storage)
)

transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

# ------------------------------VIDEO----------------------------------
"""reader = imageio.get_reader("Video.mp4")

with imageio.get_writer(
    "Output.mp4", fps=reader.get_meta_data()["fps"], macro_block_size=1
) as writer:
    for i, frame in enumerate(
        tqdm(reader, total=len([i for i, _ in enumerate(reader)]))
    ):
        frame = detect(net.eval(), frame, transform)
        writer.append_data(frame)"""

# ------------------------------IMAGE-----------------------------------
"""reader = imageio.get_reader("Image.jpg")

with imageio.get_writer("Output.jpg") as writer:
    for i, frame in enumerate(
        tqdm(reader, total=len([i for i, _ in enumerate(reader)]))
    ):
        frame = detect(net.eval(), frame, transform)
        writer.append_data(frame)"""
