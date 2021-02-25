import os
import subprocess
from utils import begins_with, replace_extension
from exr_to_png import convert
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
    
    def get_output_file_name(self,scene_name,spp):
        return f"{scene_name}/{scene_name}_{spp}spp_{self.name}.png"

    def render(self,input_file,scene_name,spp,time_data):
        cavalry_path = "C:/Users/Dunfan/Code/VSIDE/Cavalry/build_tmp/Release/Cavalry.exe"

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
        cavalry_path = "C:/Users/Dunfan/Code/VSIDE/Cavalry/build_tmp/Release/Cavalry.exe"

        output_file = self.get_output_file_name(scene_name,spp)
        command = [cavalry_path,"--input",input_file,"--output",f"{output_file}","--spp",f"{spp}","--integrator","rlpath"]

        result_lines = self.run_command(command)
        for line in result_lines:
            if begins_with(line,"Rendering took"):
                time = float(line.split()[2].split("ms")[0])/1000
                self.record_time(time_data,spp,time)


class Mitsuba(Renderer):
    def __init__(self):
        super().__init__()
        self.name = "mitsuba"
    
    def render(self,input_file,scene_name,spp,time_data):

        input_file = replace_extension(input_file,"xml")
        mitsuba_path = "C:/Users/Dunfan/Code/mitsuba/mitsuba2/build/dist/mitsuba.exe"

        sampler = "ldsampler"
        if spp not in [4,16,64,256,1024,4096]: #ldsampler only accepts power of 4..
            sampler = "independent"

        output_file = self.get_output_file_name(scene_name,spp)
        output_exr = replace_extension(output_file,"exr")
        command = [mitsuba_path,input_file,"--output",f"{output_exr}",f"-Dspp={spp}",f"-Dsampler={sampler}"]

        result_lines = self.run_command(command)
        for line in result_lines:
            if "Rendering finished" in line:
                time = line.split('(')[1].split()[1].split(')')[0]
                unit = time[-1]
                time = float(time[:-1])
                if(unit == 'm'):
                    time *= 60
                if(unit == 'h'):
                    time *= 3600
                self.record_time(time_data,spp,time)

        convert(output_exr,output_file)
        #os.remove(output_exr)


def render_batch(input_file,scene_name,max_spp):
    Path(scene_name).mkdir(parents=True, exist_ok=True)
    spp = 4
    time_data = {}
    time_data_path = f"{scene_name}/time_data.json"
    
    # load any previously computed time_data
    if Path(time_data_path).is_file():
        with open(time_data_path,"r") as f:
            time_data = json.loads(f.read())

    renderers = [PBRT3(), CavalryPathRenderer(),CavalryRLPathRenderer(),Mitsuba()]
    while spp <= max_spp:
        for renderer in renderers:
            renderer.render(input_file,scene_name,spp,time_data)
        spp *= 2
    
    with open(time_data_path,"w") as f:
        f.write(json.dumps(time_data,indent=2))

if __name__ == "__main__":
    input_file = sys.argv[1]
    scene_name = sys.argv[2]
    time_data = {}
    render_batch(input_file,scene_name,1024)