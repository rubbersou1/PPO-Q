import sys
from model.trainer import trainer
from model.utils import setup_training, get_config_path
import argparse
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
 
def pasrse_args():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('-cn', '--Config_Name', type=str, default="CartPole",help="Name of yaml file in config folder.")
    parser.add_argument('-se', '--Seed', type=int, default=42,help="Seed of this experiment.")
    parser.add_argument('-en', '--Experiment_Number', type=int, default=0,help="Number of this Experiment")
    parser.add_argument('-wc', '--Which_Classical', type=int, default=0,help="Which type of classical or hybrid agent")
    parser.add_argument('-np', '--Num_Processes', type=int, default=5,help="Number of processes run at same time")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # args_main = pasrse_args()
    # config_file = args_main.Config_Name
    # seed = args_main.Seed
    # num_exp = args_main.Experiment_Number
    # config_path = get_config_path(config_file)
    # try:
    #     args = setup_training(config_path)
    #     args.seed = seed
    #     args.num_exp = num_exp
    #     print("###############\n")
    #     print(f"Config:{config_file}")
    #     print(f"Seed: {args.seed}, Experiment number: {args.num_exp}")
    #     print("###############\n")

    # except FileNotFoundError:
    #     print(f"Error: Config file '{config_file}.yaml' not found in 'config/' directory.")

    # trainer(args)
    # print(f"exp number {args.num_exp} end")


    args_main = pasrse_args()
    config_file = args_main.Config_Name
    seeds = np.linspace(0,900,10,dtype=int) + args_main.Seed
    num_processes = args_main.Num_Processes
    num_exps = range(10)
    config_path = get_config_path(config_file)
    wc = args_main.Which_Classical
    # try: 
    #     base_args = setup_training(config_path)
    #     base_args.which_classical = wc
    #     print(f"Config:{config_file}")
    #     args_list = []
    #     for i in num_exps:
    #         args = base_args.copy() if hasattr(base_args, 'copy') else base_args 
    #         args.seed = int(seeds[i])
    #         args.num_exp = int(num_exps[i])
    #         args_list.append(args)
        
    #     mp.set_start_method('spawn', force=True)
    #     pool = mp.Pool(processes=min(num_processes, len(args_list)))
    #     with tqdm(total=len(args_list), desc="Running parallel experiments") as pbar:
    #         worker = trainer
    #         for _ in pool.imap_unordered(worker, args_list):
    #             pbar.update()

    # except FileNotFoundError:
    #     print(f"Error: Config file '{config_file}.yaml' not found in 'config/' directory.")

    # finally:
    #     pool.close()
    #     pool.join()

    try: 
        args = setup_training(config_path)
        args.which_classical = wc
        print(f"Config:{config_file}")
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}.yaml' not found in 'config/' directory.")

    for i in tqdm(num_exps):
        args.seed = int(seeds[i])
        args.num_exp = int(num_exps[i])
        print("###############\n")
        print(f"Seed: {args.seed}, Experiment number: {args.num_exp}")
        print("###############\n")
        trainer(args)



