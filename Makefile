# Copyright 2020 Xilinx
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<zcu102_base_dfx/xilinx_u50_xdma_201920_1> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to pick the specific files and generates the design for specified Target and Device."
	$(ECHO) "	   By default, HOST_ARCH=aarch64. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make exe TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "       Command to build host application."
	$(ECHO) ""
	$(ECHO) "  make bin TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "        Command to build xclbin(PL Bitstream) application."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."

# Points to top directory of Git repository
COMMON_REPO = ./lib/
PWD = $(shell readlink -f .)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

#Necessary variables
TARGET := hw
HOST_ARCH := aarch64
SYSROOT := /opt/Xilinx/petalinux2019_2/sysroot/sysroots/aarch64-xilinx-linux/
DEVICE := zcu102_base_dfx
XCLBIN := xclbin
KERNEL_NAME := hwKernels
KERNEL_NAME_1 := integrateKernel
KERNEL_NAME_2 := bilateralFilterKernel
KERNEL_NAME_3 := trackKernel
SDCARD := sd_card
CONFIG_FILE := design.cfg

# Enable Profiling
REPORT := no
PROFILE:= no

# The C++ Compiler to use is included here, depending arch
include ./utils.mk

XSA := $(call device2xsa, $(DEVICE))
BUILD_DIR := ./build/build_dir.$(TARGET).$(XSA)
BUILD_DIR_hwKernels = $(BUILD_DIR)/$(KERNEL_NAME)

# The kernel Compiler to use : V++
VPP := v++

#Include Libraries
include $(ABS_COMMON_REPO)/common/includes/opencl/opencl.mk
include $(ABS_COMMON_REPO)/common/includes/xcl2/xcl2.mk

# The below are compile flags are passed to the C++ Compiler
thirdparty_FLAGS += -I lib/TooN/include -I lib/include -I lib/thirdparty
CXXFLAGS += -O3 -g -std=c++11 -fopenmp -fpermissive -fmessage-length=0
# Thirparty flags
CXXFLAGS += $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS) $(thirdparty_FLAGS)

# The below are linking flags for C++ Compiler
LDFLAGS += $(opencl_LDFLAGS) $(xcl2_LDFLAGS)

# Host compiler global settings
LDFLAGS += -lrt -lstdc++
ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

# Kernel compiler global settings Please note that --jobs N = CPU Threads
CLFLAGS_NOLINK +=-t $(TARGET) --jobs 16 --platform $(DEVICE) --save-temps -I lib/TooN/include -I lib/include -I lib/thirdparty \
--advanced.prop kernel.integrateKernel.kernel_flags="-std=c++0x -fexceptions " -g \
--advanced.prop kernel.bilateralFilterKernel.kernel_flags="-std=c++0x -fexceptions " \
--advanced.prop kernel.trackKernel.kernel_flags="-std=c++0x -fexceptions "

CLFLAGS += -t $(TARGET) --jobs 16 --platform $(DEVICE) --config $(CONFIG_FILE) --save-temps -I lib/TooN/include -I lib/include -I lib/thirdparty
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif
CLFLAGS +=  --advanced.prop kernel.integrateKernel.kernel_flags="-std=c++0x -fexceptions"


#Host CPP FILES
HOST_CPP_SRCS += host_src/host.cpp
HOST_CPP_SRCS += host_src/kernels.cpp
HOST_CPP_SRCS += $(xcl2_SRCS)

# KFusion executable arguments
OUTPUT_FILE := benchmark.log
INPUT_FILE := living_room_traj2_loop.raw
EXECUTABLE = host
TRAJ_PARAMS := -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 1 -k 481.2,480,320,240
CMD_ARGS =  -i $(INPUT_FILE) -o $(OUTPUT_FILE) $(TRAJ_PARAMS) -x $(BINARY_CONTAINERS)
EMCONFIG_DIR = $(XCLBIN)
EMU_DIR = $(SDCARD)/data/emulation


BINARY_CONTAINERS += $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin
BINARY_CONTAINER_integrateKernel_OBJS += $(XCLBIN)/$(KERNEL_NAME_1).$(TARGET).xo
BINARY_CONTAINER_bilateralFilterKernel_OBJS += $(XCLBIN)/$(KERNEL_NAME_2).$(TARGET).xo
BINARY_CONTAINER_trackKernel_OBJS += $(XCLBIN)/$(KERNEL_NAME_3).$(TARGET).xo

CP = cp -rf
XCLBINITUTIL = xclbinutil


.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig Makefile

.PHONY: exe
exe: $(EXECUTABLE)

.PHONY: bin
bin: $(BINARY_CONTAINERS)

# Building kernel
$(BINARY_CONTAINER_integrateKernel_OBJS): hls_source/integrateKernel.cpp
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS_NOLINK) --temp_dir $(BUILD_DIR_hwKernels) -c -k $(KERNEL_NAME_1) -I'$(<D)' -o'$@' '$<'
$(BINARY_CONTAINER_bilateralFilterKernel_OBJS): hls_source/bilateralFilterKernel.cpp
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS_NOLINK) --temp_dir $(BUILD_DIR_hwKernels) -c -k $(KERNEL_NAME_2) -I'$(<D)' -o'$@' '$<'
$(BINARY_CONTAINER_trackKernel_OBJS): hls_source/trackKernel.cpp
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS_NOLINK) --temp_dir $(BUILD_DIR_hwKernels) -c -k $(KERNEL_NAME_3) -I'$(<D)' -o'$@' '$<'
$(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin: $(BINARY_CONTAINER_integrateKernel_OBJS) $(BINARY_CONTAINER_bilateralFilterKernel_OBJS) $(BINARY_CONTAINER_trackKernel_OBJS)
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR_hwKernels) -l $(LDCLFLAGS) -o'$@' $(+)
ifneq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	mkdir -p reports
	$(CP) $(BUILD_DIR_hwKernels)/reports/* ./reports/
	$(CP) $(BUILD_DIR_hwKernels)/reports/link/imp/kernel_util_synthed.rpt ./

endif


# Building Host
$(EXECUTABLE): $(HOST_CPP_SRCS)
	$(CXX) $(CXXFLAGS) $(HOST_CPP_SRCS) -o '$@' $(LDFLAGS)
	$(CP) $(EXECUTABLE) $(XCLBIN)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: bin exe emconfig
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	export XCL_EMULATION_MODE=$(TARGET) &&	./$(EXECUTABLE) $(CMD_ARGS)
else
	mkdir -p $(EMU_DIR)
	$(CP) $(XILINX_VITIS)/data/emulation/unified $(EMU_DIR)
	mkfatimg $(SDCARD) $(SDCARD).img 500000
	launch_emulator -no-reboot -runtime ocl -t $(TARGET) -sd-card-image $(SDCARD).img -device-family $(DEV_FAM)
endif
else
ifeq ($(HOST_ARCH), x86)
	 ./$(EXECUTABLE) $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin
endif
endif
ifeq ($(HOST_ARCH), x86)
ifeq ($(PROFILE), yes)
	perf_analyze profile -i profile_summary.csv -f html
endif
endif

sd_card: $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig
ifneq ($(HOST_ARCH), x86)
	mkdir -p $(SDCARD)/$(BUILD_DIR)
	$(CP) $(B_NAME)/sw/$(XSA)/boot/generic.readme $(B_NAME)/sw/$(XSA)/xrt/image/* xrt.ini $(EXECUTABLE) $(SDCARD)
	$(CP) $(BUILD_DIR)/*.xclbin $(SDCARD)/$(BUILD_DIR)/
	$(CP) data $(SDCARD)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(ECHO) 'cd /mnt/' >> $(SDCARD)/init.sh
	$(ECHO) 'export XILINX_VITIS=$$PWD' >> $(SDCARD)/init.sh
	$(ECHO) 'export XCL_EMULATION_MODE=$(TARGET)' >> $(SDCARD)/init.sh
	$(ECHO) './$(EXECUTABLE) $(CMD_ARGS)' >> $(SDCARD)/init.sh
	$(ECHO) 'reboot' >> $(SDCARD)/init.sh
else
	[ -f $(SDCARD)/BOOT.BIN ] && echo "INFO: BOOT.BIN already exists" || $(CP) $(BUILD_DIR)/sd_card/BOOT.BIN $(SDCARD)/
	$(ECHO) 'cd /mnt/' >> $(SDCARD)/init.sh
	$(ECHO) './$(EXECUTABLE) $(CMD_ARGS)' >> $(SDCARD)/init.sh
endif
endif


# Cleaning stuff
RMDIR = rm -rf

clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*, *hw*}
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	-$(RMDIR) host_src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) run.sh

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) _x.* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin

check_xrt:
ifndef XILINX_XRT
	$(error XILINX_XRT variable is not set, please set correctly and rerun)
endif


ECHO := @echo

#'estimate' for estimate report generation
#'system' for system report generation
ifneq ($(REPORT), no)
CLFLAGS += --report_level estimate
CLLDFLAGS += --report_level system
endif

#Generates profile summary report
ifeq ($(PROFILE), yes)
LDCLFLAGS += --profile_kernel data:all:all:all:all
LDCFLAGS += --profile_kernel  stall:all:all:all:all
LDCFALGS += --profile_kernel exec:all:all:all:all
endif
