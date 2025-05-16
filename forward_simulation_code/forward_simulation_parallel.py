import torch
import numpy as np
import json
import os
from scipy.io import loadmat
import forward as fd
import torch.multiprocessing as mp
import re
import torch.nn.functional as F
import time

# Load configuration and dataset mappings
with open('dataset_config2.json') as f:
    ctx = json.load(f)

datasets = {
    'PRSU': "prostate",
    'PRSD': "prostate_down"
}

def simulate_task(params):
    t0 = time.time()
    patient, angle, i, geometry, reverse, pid = params
    print(patient, angle, i, geometry, reverse, pid)
    s_down_ratio = 3
    t_down_ratio = 10
    file_path = "/work/nvme/beej/hwang39/jhu_data"#"/lustre/scratch5/ylin/jhu_data"
    # Determine GPU assignment based on the process name
    proc_name = mp.current_process().name  # e.g. "ForkPoolWorker-1"
    nums = re.findall(r'\d+', proc_name)
    worker_id = int(nums[-1]) - 1 if nums else 0
    num_gpus = torch.cuda.device_count()
    gpu_id = worker_id % num_gpus
    device = torch.device(f'cuda:{gpu_id}')
    save_path = "/work/nvme/behq/hwang39/jhu_data"#"/lustre/scratch5/ylin/jhu_data/sim2"
    os.makedirs(save_path, exist_ok=True)
    # Build the .mat filename and load the data
    label = np.zeros((15, 1, 401, 161))
    count = 0
    #for i in [0, 8, 16, 24]:#4
    for j in [0, -5, -10, -15, -20]:#5
        for k in [-10, 0, 10]:#3
            mat_file = f'{file_path}/nips_all_sos_mat/nips_sos_mat_{pid}/{patient}_projection_angle_{angle}_i_{i}_j_{j}_k_{k}_sos2d.mat'
            #a=loadmat(mat_file)['sosi']
            #print("a shape", a.shape)
            label[count,0,:,:] = loadmat(mat_file)['sosi']
            count += 1
            # print(mat_file)
            # label = loadmat(mat_file)['sosi']
    np.save(f"{save_path}/nips_all_sos_npy/{pid}/{patient}_projection_angle_{angle}_i_{i}_sos2d.npy", np.float32(label))
    if reverse:
        label = np.flip(label, axis=2)
    with torch.no_grad():
        # Run the forward simulation on the assigned GPU
        forward_model = fd.FWIForward(ctx[datasets[geometry]], device, normalize=False)
        input_tensor = torch.from_numpy(np.float32(label))
        upsampled = F.interpolate(
            input_tensor,
            scale_factor=(s_down_ratio, s_down_ratio),
            mode='bilinear',        # or 'nearest' if you want piecewise-constant
            align_corners=False     # only for 'bilinear'/'bicubic'
            )
        #print('upsampled vel shape', upsampled.shape)
        seis = forward_model(upsampled.to(device))
        seis_np = seis.cpu().detach().numpy()
    #print('simulation seis shape',seis_np.shape)
    # Save the ultrasound simulation output
    if reverse:
        if geometry == 'PRSU':
            np.save(f'{save_path}/nips_downsample_data/{pid}/{patient}_projection_angle_{angle}_i_{i}_{geometry}_sos2d_reverse.npy', np.float32(seis_np)[:,:,::t_down_ratio,::s_down_ratio])
            #print("saved file size", np.float32(seis_np)[:,:,::t_down_ratio,::s_down_ratio].shape, seis_np.min(), seis_np.max())
        elif geometry == 'PRSD':
            np.save(f'{save_path}/nips_downsample_data/{pid}/{patient}_projection_angle_{angle}_i_{i}_{geometry}_sos2d_reverse.npy', np.float32(seis_np[:,:,5000:,:])[:,:,::t_down_ratio,::s_down_ratio])
            #print("saved file size", np.float32(seis_np[:,:,5000:,:])[:,:,::t_down_ratio,::s_down_ratio].shape)
    else:
        if geometry == 'PRSU':
            np.save(f'{save_path}/nips_downsample_data/{pid}/{patient}_projection_angle_{angle}_i_{i}_{geometry}_sos2d.npy', np.float32(seis_np)[:,:,::t_down_ratio,::s_down_ratio])
            #print("saved file size", np.float32(seis_np)[:,:,::t_down_ratio,::s_down_ratio].shape, seis_np.min(), seis_np.max())
        elif geometry == 'PRSD':
            np.save(f'{save_path}/nips_downsample_data/{pid}/{patient}_projection_angle_{angle}_i_{i}_{geometry}_sos2d.npy', np.float32(seis_np[:,:,5000:,:])[:,:,::t_down_ratio,::s_down_ratio])
            #print("saved file size", np.float32(seis_np[:,:,5000:,:])[:,:,::t_down_ratio,::s_down_ratio].shape, seis_np.min(), seis_np.max())
    t1 = time.time()
    print(f"Finished simulation for {patient} (angle={angle}, i={i}, geometry={geometry}, reverse={reverse}) on GPU {gpu_id}")
    print(f"Single forward sim time: {t1 - t0:.6f} s")
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-GPU forward simulation for prostates')
    parser.add_argument('--start', default=0, type=int, help='start idx')
    parser.add_argument('-n', '--num', default=3, type=int, help='patient number')
    #parser.add_argument('--reverse', action='store_true', help='reverse')
    parser.add_argument('-pid', '--patient', default="1_01", type=str, help='patient id')

    args = parser.parse_args()
    pid = args.patient
    # Define your patients list.
    # For example, if your patients are named like "3_01_P_2021-04-20", "3_01_P_2021-04-22", etc.
    patients = [f"{args.patient}_P_2021-03-16",f"{args.patient}_P_2021-03-29",f"{args.patient}_P_2021-03-30",f"{args.patient}_P_2021-04-08",f"{args.patient}_P_2021-04-19",
                f"{args.patient}_P_2021-04-20",f"{args.patient}_P_2021-04-22",f"{args.patient}_P_2021-04-27",f"{args.patient}_P_2021-05-04",f"{args.patient}_P_2021-05-06",
                f"{args.patient}_P_2021-05-17",f"{args.patient}_P_2021-05-18",f"{args.patient}_P_2021-05-24",f"{args.patient}_P_2021-06-15",f"{args.patient}_P_2021-06-18",
                f"{args.patient}_P_2021-06-22",f"{args.patient}_P_2021-06-25",f"{args.patient}_P_2021-07-13",f"{args.patient}_P_2021-07-22",f"{args.patient}_P_2021-07-26",
                f"{args.patient}_P_2021-07-27",f"{args.patient}_P_2021-07-29",f"{args.patient}_P_2021-07-30",f"{args.patient}_P_2021-08-05",f"{args.patient}_P_2021-08-19",
                f"{args.patient}_P_2021-08-31",f"{args.patient}_P_2021-09-02",f"{args.patient}_P_2021-09-07",f"{args.patient}_P_2021-09-14",f"{args.patient}_P_2021-09-16",
                f"{args.patient}_P_2021-09-20",f"{args.patient}_P_2021-09-21",f"{args.patient}_P_2021-09-28",f"{args.patient}_P_2021-09-30",f"{args.patient}_P_2021-10-07",
                f"{args.patient}_P_2021-10-26",f"{args.patient}_P_2021-11-02",f"{args.patient}_P_2021-11-16",f"{args.patient}_P_2021-11-30",f"{args.patient}_P_2021-12-06",
                f"{args.patient}_P_2021-12-17",f"{args.patient}_P_2022-01-13",f"{args.patient}_P_2022-01-24",f"{args.patient}_P_2022-02-07",f"{args.patient}_P_2022-02-15",
                f"{args.patient}_P_2022-02-23",f"{args.patient}_P_2022-03-01",f"{args.patient}_P_2022-03-14",f"{args.patient}_P_2022-03-28",f"{args.patient}_P_2022-03-29",
                f"{args.patient}_P_2022-04-11",f"{args.patient}_P_2022-04-18",f"{args.patient}_P_2022-04-19",f"{args.patient}_P_2022-05-03",f"{args.patient}_P_2022-05-05",
                f"{args.patient}_P_2022-05-23",f"{args.patient}_P_2022-05-24",f"{args.patient}_P_2022-05-26",f"{args.patient}_P_2022-05-31",f"{args.patient}_P_2022-06-02",
                f"{args.patient}_P_2022-06-06",f"{args.patient}_P_2022-06-09"]

    # Build a list of tasks for the selected patients
    tasks = []
    for patient in patients[args.start: args.start + args.num]:
        for angle in [45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45]:#19
      # for angle in [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
            for i in [0, 8, 16, 24]:#4
            #     for j in [0, -5, -10, -15, -20]:#5
            #         for k in [-10, 0, 10]:#3
                for geometry in ['PRSU','PRSD']:
                    for reverse in [True, False]:
                        tasks.append((patient, angle, i, geometry, reverse, args.patient))

    # Set the multiprocessing start method and run tasks concurrently
    mp.set_start_method('spawn', force=True)
    num_processes = torch.cuda.device_count()  # ideally one process per GPU
    print(f"Running simulation on {num_processes} processes (GPUs)")
    with mp.Pool(processes=num_processes) as pool:
        pool.map(simulate_task, tasks)


