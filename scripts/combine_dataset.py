import zarr
import time
import numpy as np
import click
import pathlib

@click.command()
@click.option("-o","--output",default="./combined_data",help="target output directory")
@click.option("-f","--file",multiple=True,required=True,type=click.Path(exists=True),help="input files(just like --file file1 --file file2)")
def main(output:str,file:tuple):
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    zrootname = "recorded_data_{}_{}.zarr".format('combined', time.strftime("%H-%M-%S", time.localtime()))
    zroot = zarr.open_group(output+"/"+zrootname,mode='w')
    zroot.create_group("data")
    zdata = zroot["data"]

    zroot.create_group("meta")
    zmeta = zroot["meta"]

    zmeta.create_group("episode_ends")

    zdata.create_group("action")
    zdata.create_group("state")

    action = np.zeros((0,12),dtype=np.float64)
    state = np.zeros((0,45),dtype=np.float64)
    episode_ends = np.zeros((0,),dtype=np.int64) 

    episode_end:int = 0
    for data_source in file:
        print(f"Processing {data_source}")
        zfile = zarr.open(data_source,"r")
        action = np.concatenate([action, zfile["data"]["action"]])
        state = np.concatenate([state, zfile["data"]["state"]])
        episode_ends = np.concatenate([episode_ends, np.array(zfile["meta"]["episode_ends"]) + episode_end])
        episode_end = episode_end + zfile["meta"]["episode_ends"][-1]

    zdata["action"] = action
    zdata["state"] = state
    zmeta["episode_ends"] = episode_ends
        
if __name__ == "__main__":
    main()