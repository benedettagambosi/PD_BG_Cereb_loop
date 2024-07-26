## Testing conditions
Ubuntu 22.04.1 LTS (GNU/Linux 5.15.0-112-generic x86_64)

Python version 3.10.12 without virtual environments

`$HOME` is `/home/ubuntu`

## Install NEST version 2.20.2

```
wget https://zenodo.org/records/5242954/files/nest-simulator-2.20.2.tar.gz
tar -xzvf nest-simulator-2.20.2.tar.gz
mkdir nest-simulator-build
mkdir nest-simulator-install
cd nest-simulator-build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/home/ubuntu/nest-simulator-2.20.2-install/ /home/ubuntu/nest-simulator-2.20.2
make -j16
make install
```

Then modify the `.bashrc` script, e.g. with `nano`
```
cd
nano .bashrc
```

Add at the bottom of the file
```
source /home/ubuntu/nest-simulator-2.20.2-install/bin/nest_vars.sh
```

Then save, exit and source it

```
source .bashrc
```

run `python3` and then `import nest` if you see something like:

```
>>> import nest
[INFO] [2024.7.10 15:44:32 /home/ubuntu/nest-simulator-2.20.2/nestkernel/rng_manager.cpp:217 @ Network::create_rngs_] : Creating default RNGs
[INFO] [2024.7.10 15:44:32 /home/ubuntu/nest-simulator-2.20.2/nestkernel/rng_manager.cpp:260 @ Network::create_grng_] : Creating new default global RNG

              -- N E S T --
  Copyright (C) 2004 The NEST Initiative

 Version: nest-2.20.2
 Built: Jul 10 2024 15:07:13

 This program is provided AS IS and comes with
 NO WARRANTY. See the file LICENSE for details.

 Problems or suggestions?
   Visit https://www.nest-simulator.org

 Type 'nest.help()' to find out more about NEST.

>>> 
```
You have succesfully installed NEST

## Install Cereb-nest module

```
wget https://zenodo.org/records/10970203/files/cereb-nest-nest2.20.2_python3.8.zip
unzip cereb-nest-nest2.20.2_python3.8.zip
```

check if `echo $NEST_INSTALL_DIR` gives you the correct path: `/home/ubuntu/nest-simulator-2.20.2-install`

```
mkdir cereb-nest-build
cd cereb-nest-build
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config /home/ubuntu/cereb-nest-nest2.20.2_python3.8
make
make install
cd
```

run `python3`, then `import nest`, then `nest.Install("cerebmodule")`  if you see something like:
```
>>> nest.Install("cerebmodule")

Jul 10 15:55:23 Install [Info]: 
    loaded module Cereb Module
```
You have succesfully installed the cereb-nest module

## Install bgmodel
```
wget https://zenodo.org/records/10970203/files/bgmodel-master.zip
unzip bgmodel-master.zip
cd bgmodel-master
./install-module-2.20.2.sg $NEST_INSTALL_DIR
```

run `python3`, then `import nest`, then `nest.Install("ml_module")`  if you see something like:
```
>>> nest.Install("ml_module")

Jul 10 16:15:05 Install [Info]: 
    loaded module Ml NEST Module
```
You have succesfully installed the bgmodel module

## Install other libraries
`pip install fooof==1.0.0`

## Download the ZENODO archive and run simulations
```
wget https://zenodo.org/records/10970203/files/PD_BG_Cereb_loop.zip
unzip PD_BG_Cereb_loop
cd PD_BG_Cereb_loop
```

here you can:

* Run `main.py` to perform a simulation

* Run `run_script.py` to perform multiple times main.py

See [README.md](https://github.com/benedettagambosi/PD_BG_Cereb_loop/blob/main/README.md) for instructions.
