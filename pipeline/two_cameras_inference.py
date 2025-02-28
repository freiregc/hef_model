import degirum as dg
import degirum_tools
import degirum_tools.streams as dgstreams

import os

# Limitar o número de threads para operações paralelas
#os.environ["OMP_NUM_THREADS"] = "2"
#os.environ["OPENBLAS_NUM_THREADS"] = "2"
#os.environ["MKL_NUM_THREADS"] = "2"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
#os.environ["NUMEXPR_NUM_THREADS"] = "2"


inference_host_address = "@local"
# inference_host_address = "@local"

# choose zoo_url

#desktop
zoo_url = "/home/gabriel/Desktop/rasp/hailo-rpi5-examples/resources/best_22-01_01_i640.json"

#raspberry
#zoo_url = "/home/pi/Desktop/hailo-rpi5-examples/resources/best_22-01_01_i640.json"

# set token
#token = degirum_tools.get_token()
token = ''
# token = '' # leave empty for local inference

#webcams
source1 = 0 # Webcam index
source2 = 2 # Webcam index

#videos desktop 
source3 = "/home/gabriel/Desktop/rasp/hailo-rpi5-examples/resources/11_29_2024_11_50_00_cut1.avi"  # Video file
source4 = "/home/gabriel/Desktop/rasp/hailo-rpi5-examples/resources/11_29_2024_11_50_00_cut2.avi"  # Video file
source5 = "/home/gabriel/Desktop/rasp/hailo-rpi5-examples/resources/11_29_2024_11_50_00.avi"  # Video file

#videos rasp
#source3 = "/home/pi/Desktop/hailo-rpi5-examples/resources/11_29_2024_11_50_00_cut1.avi"  # Video file
#source4 = "/home/pi/Desktop/hailo-rpi5-examples/resources/11_29_2024_11_50_00_cut2.avi"  # Video file
#source5 = "/home/pi/Desktop/hailo-rpi5-examples/resources/11_29_2024_11_50_00.avi"  # Video file

# Define the configurations for video file and webcam
configurations = [
    {
        "model_name": "best_22-01_01_i640",
        "source":source1,
        "display_name": "Video/Cam-1",
    },
    #{
    #    "model_name": "best_22-01_01_i640",
    #    "source": source2,
    #    "display_name": "Video/Cam-2",
    #},
]

# load models
models = [
    dg.load_model(cfg["model_name"], inference_host_address, zoo_url, token)
    for cfg in configurations
]

# define gizmos
sources = [dgstreams.VideoSourceGizmo(cfg["source"]) for cfg in configurations]
detectors = [dgstreams.AiSimpleGizmo(model) for model in models]
display = dgstreams.VideoDisplayGizmo(
    [cfg["display_name"] for cfg in configurations], show_ai_overlay=True, show_fps=True
)

# create pipeline
pipeline = (
    (source >> detector for source, detector in zip(sources, detectors)),
    (detector >> display[di] for di, detector in enumerate(detectors)),
)

# start composition
dgstreams.Composition(*pipeline).start()
