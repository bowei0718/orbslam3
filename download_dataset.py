# from roboflow import Roboflow
# rf = Roboflow(api_key="ZaWZscFTeQdbGPVovpQT")
# project = rf.workspace("0621-dsqyp").project("2026_0409_asrlab215")
# version = project.version(1)
# dataset = version.download("yolov11")

from roboflow import Roboflow
rf = Roboflow(api_key="ZaWZscFTeQdbGPVovpQT")
project = rf.workspace("0621-dsqyp").project("2026_0409_asrlab215")
version = project.version(2)
dataset = version.download("yolov11")              
