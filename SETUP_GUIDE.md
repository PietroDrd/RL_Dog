# Ubuntu22.04 + Microconda + IsaacLAB + Pytorch + SKRL

I have developed the project in Ubuntu 22.04, however is it possible to do it in Windows (10/11) too

For the following setup-tutorial I am supposing that you are already working in Ubuntu, if not consider to install it in dual-boot or in a dedicated machine: [Ubuntu Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)


## Basic commands:

```
sudo apt update
sudo apt upgrade -y
sudo apt dist-upgrade -y
sudo apt autoremove -y
sudo apt autoclean
sudo apt clean
```

## Install Miniconda (or Anaconda) on linux:  [Guide](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)

These four commands quickly and quietly install the latest 64-bit version of the installer and then clean up after themselves.
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

```
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

## Install Fuse
```
sudo apt install -y fuse 
```

## INSTALL ISAAC SIM [DOC](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)

Pip install does not work, use **Pre-Built Binary** !

go to [Omniverse_Install](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) and download omniverse launcher

```
cd Downloads/
chmod +x omniverse-launcher-linux.AppImage

./omniverse-launcher-linux.AppImage
```
install IsaacSim (9GB) and wait
then you can close and proceed with IsaacLab installation


## INSTALL ISAAC LAB: [Guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html)

```
git clone https://github.com/isaac-sim/IsaacLab.git

cd IsaacLab

ln -s #####path_to_isaac_sim (isaac-sim-4.0.0)##### _isaac_sim

./isaaclab.sh --conda isaacenv # Name of env is isaacenv
```

```
conda activate isaacenv

sudo apt install cmake build-essential

./isaaclab.sh --install skrl
```


Other Conda installations (now  that "isaacenv" is active)
```
conda install pytorch==2.2.2 cudatoolkit=11.8 -c pytorch
conda install anaconda-navigator
```
## ISAAC LAB CONFIGURATIONS

Isaac Sim root directory
```
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0"
```

Isaac Sim python executable
```
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

Check that the simulator runs as expected:

```
${ISAACSIM_PATH}/isaac-sim.sh
```

<br>

Check that the simulator runs from a standalone python script:
```
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"

${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/ standalone_examples/api/omni.isaac.core/add_cubes.py
```

<br>

Verify IsaacLab Installation  

```
# Option 1: Using the isaaclab.sh executable
# note: this works for both the bundled python and the virtual environment
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

# Option 2: Using python in your virtual environment
python source/standalone/tutorials/00_sim/create_empty.py
```