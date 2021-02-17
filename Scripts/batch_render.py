import os
import subprocess
from utils import begins_with
import sys
from pathlib import Path
import json

class Renderer:
    name:str

    def get_output_file_name(self,scene_name,spp):
        return f"{scene_name}/{scene_name}_{spp}spp_{self.name}.png"

    def record_time(self, time_data,spp,time):
        if self.name not in time_data:
            time_data[self.name] = {}
        spp_key = f"{spp}spp"
        time_data[self.name][spp_key] = time

    def run_command(self,command):
        lines = []
        proc = subprocess.Popen(command,stdout=subprocess.PIPE)
        print(f"Running command: {' '.join(command)}")
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.rstrip().decode()
            print(line)
            lines.append(line)
        return lines
        
        

class PBRT3(Renderer):
    def __init__(self):
        super().__init__()
        self.name = "pbrt3"

    def render(self,input_file,scene_name,spp,time_data):
        # a custom version of pbrt which supports the --output and --spp command line arguments, and has ProgressReporter modified for better piping
        pbrt_path = "C:/Users/Dunfan/Code/pbrt/pbrt-v3/build/Release/pbrt.exe"

        output_file = self.get_output_file_name(scene_name,spp)
        command = [pbrt_path,input_file,"--output",f"{output_file}","--spp",f"{spp}"]

        result_lines = self.run_command(command)
        for line in result_lines:
            if begins_with(line,"Total time"):
                time = float(line.split()[2].split("s")[0])
                self.record_time(time_data,spp,time)


class CavalryPathRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.name = "path"

    def render(self,input_file,scene_name,spp,time_data):
        cavalry_path = cavalry_path = "C:/Users/Dunfan/Code/VSIDE/Cavalry/build_tmp/Release/Cavalry.exe"

        output_file = self.get_output_file_name(scene_name,spp)
        command = [cavalry_path,"--input",input_file,"--output",f"{output_file}","--spp",f"{spp}","--integrator","path"]

        result_lines = self.run_command(command)
        for line in result_lines:
            if begins_with(line,"Rendering took"):
                time = float(line.split()[2].split("ms")[0])/1000
                self.record_time(time_data,spp,time)

class CavalryRLPathRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.name = "rlpath"

    def render(self,input_file,scene_name,spp,time_data):
        cavalry_path = cavalry_path = "C:/Users/Dunfan/Code/VSIDE/Cavalry/build_tmp/Release/Cavalry.exe"

        output_file = self.get_output_file_name(scene_name,spp)
        command = [cavalry_path,"--input",input_file,"--output",f"{output_file}","--spp",f"{spp}","--integrator","rlpath"]

        result_lines = self.run_command(command)
        for line in result_lines:
            if begins_with(line,"Rendering took"):
                time = float(line.split()[2].split("ms")[0])/1000
                self.record_time(time_data,spp,time)


def render_batch(input_file,scene_name,max_spp):
    Path(scene_name).mkdir(parents=True, exist_ok=True)
    spp = 1
    time_data = {}
    renderers = [PBRT3(), CavalryPathRenderer(),CavalryRLPathRenderer()]
    while spp <= max_spp:
        for renderer in renderers:
            renderer.render(input_file,scene_name,spp,time_data)
        spp *= 2
    
    time_data_path = f"{scene_name}/time_data.json"
    with open(time_data_path,"w") as f:
        f.write(json.dumps(time_data))

if __name__ == "__main__":
    input_file = sys.argv[1]
    scene_name = sys.argv[2]
    time_data = {}
    render_batch(input_file,scene_name,1)