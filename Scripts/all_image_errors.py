from image_error import mse
import json
import sys
import cv2

def image_file_name(scene_name,spp,renderer):
    return f"{scene_name}/{scene_name}_{spp}spp_{renderer}.png"

def compute_all_errors(scene_name):
    cv_flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    error_data = {}
    with open(f"{scene_name}/time_data.json","r") as f:
        time_data = json.loads(f.read())
    for renderer in time_data:
        if renderer not in error_data:
            error_data[renderer] = {}
        max_spp = 0
        for spp_str in time_data[renderer]:
            spp = int(spp_str.split('spp')[0])
            max_spp = max(max_spp,spp)
        max_img = cv2.imread(image_file_name(scene_name,max_spp,renderer), cv_flags).astype("float32")
        for spp_str in time_data[renderer]:
            spp = int(spp_str.split('spp')[0])
            img = cv2.imread(image_file_name(scene_name,spp,renderer), cv_flags).astype("float32")
            error = mse(max_img,img)
            error_data[renderer][spp_str] = float(error)
    with open(f"{scene_name}/error_data.json","w") as f:
        f.write(json.dumps(error_data,indent=2))

if __name__ == "__main__":
    compute_all_errors(sys.argv[1])