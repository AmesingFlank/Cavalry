import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import math

def read_data(scene_name):
    with open(f"{scene_name}/time_data.json","r") as f:
        time_data = json.loads(f.read())
    with open(f"{scene_name}/error_data.json","r") as f:
        error_data = json.loads(f.read())
    return (time_data,error_data)

def plot_error_vs_time(time_data,error_data,scene_name):
    fig, ax = plt.subplots()

    for renderer in ['path','rlpath','pbrt3','mitsuba']:
        if renderer not in error_data or renderer not in time_data:
            print(f"warning! renderer {renderer} not found")
        times = []
        errors = []
        for spp_str in error_data[renderer]:
            time = time_data[renderer][spp_str]
            error = error_data[renderer][spp_str]
            times.append(time)
            errors.append(error)
        #sort using times as key
        errors = [e for t,e in sorted(zip(times,errors))]
        times = [t for t,e in sorted(zip(times,errors))]

        errors = errors[:-1] # exclude final item, which is used as reference image
        times = times[:-1]
        
        ax.plot(times, errors,marker='o',markersize=4, label=renderer) 

    ax.set_xscale("log",basex=2)
    ax.set_yscale("log",basey=2)

    ax.set_xlabel('time/s') 
    ax.set_ylabel('MSE')  
    ax.set_title("MSE vs Rendering Time") 
    ax.legend()  # Add a legend.
    fig.savefig(f"{scene_name}/error_vs_time.png")

    

def plot_all(scene_name):
    (time_data,error_data) = read_data(scene_name)
    plot_error_vs_time(time_data,error_data,scene_name)

if __name__ == "__main__":
    plot_all(sys.argv[1])
