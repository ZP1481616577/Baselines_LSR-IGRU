import argparse

import qlib
import ruamel.yaml as yaml
from qlib.utils import init_instance_by_config
import os,sys
os.chdir(sys.path[0]) #使用文件所在目录

def main(seed, config_file="configs/config_transformer_tra.yaml"):

    # set random seed
    with open(config_file) as f:
        config = yaml.safe_load(f)  #返回python字典

    # seed_suffix = "/seed1000" if "init" in config_file else f"/seed{seed}"
    seed_suffix = ""
    config["task"]["model"]["kwargs"].update(
        {"seed": seed, "logdir": config["task"]["model"]["kwargs"]["logdir"] + seed_suffix}
    )

    # initialize workflow
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],

    )
    dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    # train model
    model.fit(dataset)



if __name__ == "__main__":

    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/config_transformer_tra.yaml", help="config file")
    args = parser.parse_args() #通过args = parser.parse_args()把刚才的属性从parser给args，后面直接通过args使用
    main(**vars(args))
