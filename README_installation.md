## Testing conditions
Ubuntu 22.04.1 LTS (GNU/Linux 5.15.0-112-generic x86_64)

Python version 3.10.12 without virtual environments

`$HOME` is `/home/ubuntu`

## Install NEST version 3.5

```
download source code (tar.gz) https://github.com/nest/nest-simulator/releases/tag/v3.5
tar -xzvf nest-simulator-3.5.tar.gz
and follow these instructions https://nest-simulator.readthedocs.io/en/v3.5/installation/noenv_install.html#noenv
```


## Install modules

```
unzip cereb_nest3_5.zip

unzip bg_nest3_5.zip
```
```
MYMODULES_DIR=$PACKAGES/tvb-multiscale/tvb_multiscale/tvb_nest/nest/modules
MYMODULES_BLD_DIR=$BUILD/nest_modules_builds
cp -r ${MYMODULES_DIR} ${MYMODULES_BLD_DIR}
MYMODULES_LIST="cereb_benny"
MYMODULES_DIR=$PACKAGES/tvb-multiscale/tvb_multiscale/tvb_nest/nest/modules
NEST_CONFIG=${NEST_INSTALL_DIR}/bin/nest-config
for MYMODULE_NAME in $MYMODULES_LIST; do export MYMODULE_DIR=${MYMODULES_DIR}/${MYMODULE_NAME}; \
        export MYMODULE_BLD=${MYMODULES_BLD_DIR}/${MYMODULE_NAME}module_bld; \
        mkdir -p ${MYMODULE_BLD}; cd ${MYMODULE_BLD}; \
            cmake -Dwith-nest=$NEST_CONFIG ${MYMODULE_DIR}; \
            make; make install; \
    done
```
## Install other libraries
`pip install fooof==1.0.0`

