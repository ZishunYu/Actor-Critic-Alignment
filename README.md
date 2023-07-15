# Actor-Critic Alignment (ACA)

Code of our paper [Actor-Critic Alignment for Offline-to-Online Reinforcement Learning](https://proceedings.mlr.press/v202/yu23k.html)

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
1. Download offline pretrained models from [here (Google drive)](https://drive.google.com/file/d/16UelW3f_N-p57dhEu5slGkcwbgNwZFoG/view?usp=sharing)
2. Run experiment with
   ~~~
   python3 run_aca.py --dataset hopper-medium-v2 --seed 1
   ~~~

## Reference
<div id="user-content-toc">
  <ul>

~~~
@InProceedings{pmlr-v202-yu23k,
  title = 	 {Actor-Critic Alignment for Offline-to-Online Reinforcement Learning},
  author =       {Yu, Zishun and Zhang, Xinhua},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {40452--40474},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/yu23k/yu23k.pdf},
  url = 	 {https://proceedings.mlr.press/v202/yu23k.html},
}
~~~

  </ul>
</div>



## Troubleshooting
1. MuJoCo installation troubleshooting, see [MuJoCo official git page](https://github.com/openai/mujoco-py#troubleshooting)
2. ```ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory```, try setting the lib path before running experiment
    ~~~
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/PATH/TO/CONDA/envs/ACA/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/YOUR_USER_NAME/.mujoco/mujoco200/bin
    ~~~
3. ```OSError: /some/path/mujoco/libmujoco200.so: undefined symbol: __glewBindBuffer```, try install libglfw3 and libglew2.0 by
    ~~~
    conda install -c menpo glfw3
    conda install -c conda-forge glew==2.0.0
    ~~~


