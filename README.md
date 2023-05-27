# Actor-Critic Alignment (ACA)

Code of our paper [Actor-Critic Alignment for Offline-to-Online Reinforcement Learning]()

## Installation

1. Pull this repo 
    ~~~
    git clone git@github.com:ZishunYu/ACA.git; cd ACA
    ~~~
2. Create conda virtual env
    ~~~
    conda create --name ACA python=3.7.4; conda activate ACA
    ~~~
3. Install MuJoCo**200** following the [official documentation](https://github.com/openai/mujoco-py)

4. Install [d4rl](https://github.com/rail-berkeley/d4rl)
    ~~~
    git clone https://github.com/rail-berkeley/d4rl.git
    cd d4rl; pip3 install -e .; cd ..
    ~~~
5. Install requirements
   ~~~
   pip3 install -r requirements.txt
   ~~~



## Run ACA
1. Download offline pretrained models from [here (Google drive)]()
2. Run experiment with
   ~~~
   python3 run_aca.py --dataset hopper-medium-v2 --seed 1
   ~~~

## Troubleshooting
1. MuJoCo installation troubleshooting, please see their [official git page](https://github.com/openai/mujoco-py#troubleshooting)
2. If you encounter ```ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory```, try setting the lib path before running experiment
    ~~~
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/PATH/TO/CONDA/envs/ACA/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/YOUR_USER_NAME/.mujoco/mujoco200/bin
    ~~~
3. If you encounter ```OSError: /some/path/mujoco/libmujoco200.so: undefined symbol: __glewBindBuffer```, try install libglfw3 and libglew2.0 by
    ~~~
    conda install -c menpo glfw3
    conda install -c conda-forge glew==2.0.0
    ~~~


