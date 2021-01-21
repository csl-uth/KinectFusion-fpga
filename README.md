# KinectFusion-FPGA

This repository contains our MPSoC FPGA configurations for KinectFusion SLAM (Simultaneous Localization and Mapping) algorithm. Our work introduces and evaluates a plethora of designs, featuring a variety of precise and approximate optimizations, to highlight the interplay between SLAM performance and accuracy.

Repository branches:
* sw-only: unoptimized baseline implementation running on ARM with OpenMP (```THREAD_NUMBER := 4```)
* baselineHW: unoptimized baseline HW implementation (one HW accelerator for each one of the Bilateral Filter, Tracking, Integration kernels)
* Conf1: HW implementation which uses the 5 most impactful (precise) optimizations 
* fastest_precise: HW implementation which uses all the precise optimizations
* Conf2: HW implementation which uses the 15 most impactful optimizations 
* fastest_approx: HW implementation which achieves the best performance (27.5 FPS)
## Paper

Maria Rafaela Gkeka, Alexandros Patras, Christos D. Antonopoulos, Spyros Lalis, Nikolaos Bellas. FPGA Architectures for Approximate Dense SLAM Computing. Design, Automation & Test in Europe Conference & Exhibition, (DATE), February 1-5, 2021, Grenoble, France.

## Usage
Vitis™ Unified Software Platform was used to run our experiments on Zynq UltraScale+ MPSoC ZCU102 Evaluation Kit. If you wish to build and run on an Alveo™ card you must make the necessary changes. 

### Requirements for the build environment
1. [Vitis™ Unified Software Platform 2019.2](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2019-2.html)
2. [Vitis™ Embedded Platform with DFX 2019.2 or 2020.1](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms/2019-2.html)
3. PetaLinux 2019.2 Sysroot directory for aarch32/aarch64 architectures - [Link](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-design-tools/2019-2.html)
4. Xilinx Runtime (for software emulation on x86 or Alveo™ cards) - [https://github.com/Xilinx/XRT](https://github.com/Xilinx/XRT)

Please consult Xilinx documentation for more detailed information.

### Requirements for the embedded platform
An embedded platform that runs PetaLinux with enabled XRT and OpenCL support. Check this [repository](https://github.com/Xilinx/Vitis_Embedded_Platform_Source) for more information.

### Build
There is a Makefile included that automates the build process for the ZCU102 board. It is based on makefiles provided in [Xilinx Vitis Acceleration Examples](https://github.com/Xilinx/Vitis_Accel_Examples/tree/2019.2). You can make changes to the Makefile according to your setup.

Commands to build for ZCU102:
1. `make bin DEVICE=zcu102_base_dfx`
2. `make exe HOST_ARCH=aarch64 SYSROOT=<path/to/sysroots/aarch64-xilinx-linux>`

The results are created inside `xclbin` directory. Software binary is named `host` and the hardware binary `<name>.hw.xclbin`.

### Run
For our experiments we use the living-room trajectory 2 [ICL-NUIM dataset](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html). We create the `.raw` format using the `scene2raw.cpp` program of the slambench1 repo. Depending on the input trajectory you must provide the necessary arguments. For instance, to run trajectory 2 (using the default parameters) you must type the command:

`./host -i living_room_traj2_loop.raw -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 1 -k 481.2,480,320,240 -o output.log -x hwKernels.hw.xclbin`

### Get in Touch

If you would like to ask questions, report bugs or collaborate on research projects, please email any of the following:

 - Maria Gkeka (margkeka at uth dot gr)
 - Alexandros Patras (patras at uth dot gr)

For more information for our laboratory visit [https://csl.e-ce.uth.gr](https://csl.e-ce.uth.gr) 

## License

[MIT](https://choosealicense.com/licenses/mit/)

We note that we use code from [SLAMBench1](https://github.com/pamela-project/slambench1) and [TooN library](https://github.com/edrosten/TooN) repositories, which each one retains its original license.
* slambench1 copyright MIT
* TooN copyright 2-Clause BSD (The original license file also exists on /lib/TooN) 
