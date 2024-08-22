from gradio_client import Client, handle_file
from os import path
from shutil import rmtree
from matplotlib import pyplot

HUMAN_IMG = "res/background.jpg"
CLOTHE_IMG = "res/clothes_top.jpg"

client = Client("kadirnar/IDM-VTON", download_files="./res/gradio/")
result = client.predict(
		dict={"background": handle_file(HUMAN_IMG),"layers":[],"composite":  None},
		garm_img=handle_file(CLOTHE_IMG),
		garment_des="Hello!!",
		is_checked=True,
		is_checked_crop=True,
		denoise_steps=30,
		seed=42,
		api_name="/tryon"
)

# 마스킹 이미지 제거
rmtree(path.dirname(result[1]))

img = pyplot.imread(result[0])
pyplot.imshow(img)
pyplot.show()