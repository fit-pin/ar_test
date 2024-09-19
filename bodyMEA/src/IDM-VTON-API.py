from gradio_client import Client, handle_file
from os import path
from shutil import rmtree
from matplotlib import pyplot

HUMAN_IMG = "res/test.jpg"
CLOTHE_IMG = "res/clothes_top.jpg"

client = Client("yisol/IDM-VTON", download_files="./res/gradio/")

job = client.submit(
    dict={"background": handle_file(HUMAN_IMG), "layers": [], "composite": None},
    garm_img=handle_file(CLOTHE_IMG),
    garment_des="any",
    is_checked=True,
    is_checked_crop=True,
    denoise_steps=30,
    seed=42,
    api_name="/tryon",
)

print(job.status())

result = job.result()

# 마스킹 이미지 제거
rmtree(path.dirname(result[1]))

img = pyplot.imread(result[0])
pyplot.imshow(img)
pyplot.show()
