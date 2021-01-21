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
	$(ECHO) "  make all HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to pick the specific files and generates the design for specified Target and Device."
	$(ECHO) "	   By default, HOST_ARCH=aarch64. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make check HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""	
	$(ECHO) "  make exe HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to build exe application"
	$(ECHO) ""
	$(ECHO) "  make clean 
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."

# Points to top directory of Git repository
COMMON_REPO = ./lib/
PWD = $(shell readlink -f .)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

TARGET := hw
HOST_ARCH := aarch64
SYSROOT := /opt/Xilinx/petalinux/sysroot/sysroots/aarch64-xilinx-linux/
DEVICE := zcu102_base
SDCARD := sd_card
XCLBIN := xclbin

# The C++ Compiler to use is included here, depending arch
include ./utils.mk

XSA := $(call device2xsa, $(DEVICE))


# The below are compile flags are passed to the C++ Compiler
thirdparty_FLAGS += -I lib/TooN/include -I lib/include -I lib/thirdparty 
CXXFLAGS +=  $(thirdparty_FLAGS) -Wall -O3 -g -std=c++11 -fopenmp -fmessage-length=0

# The below are linking flags for C++ Compiler
ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

#Host CPP FILES
HOST_CPP_SRCS += host_src/host.cpp
HOST_CPP_SRCS += host_src/kernels.cpp
# Host Header FILE
HOST_CPP_HDRS += lib/include/host.hpp

ifeq ($(HOST_ARCH),x86)
	CXXFLAGS += -DARCH_X86
endif


# is always enabled
OUTPUT_FILE := benchmark.log
INPUT_FILE := living_room_traj2_loop.raw
EXECUTABLE = host
TRAJ_PARAMS := -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 1 -k 481.2,480,320,240
CMD_ARGS =  -i $(INPUT_FILE) $(TRAJ_PARAMS) -o $(OUTPUT_FILE)

# NUMBER OF THREADS TO RUN
THREAD_NUMBER := 4

CP = cp -rf
SFTP = scp 
SSH = ssh


.PHONY: all clean cleanall docs emconfig
all: $(EXECUTABLE) Makefile sd_card

.PHONY: exe
exe: $(EXECUTABLE)

# Building Host
$(EXECUTABLE): $(HOST_CPP_SRCS) $(HOST_CPP_HDRS)
	$(CXX) $(CXXFLAGS) $(HOST_CPP_SRCS) $(HOST_CPP_HDRS) -o '$@' $(LDFLAGS)
	mkdir -p $(XCLBIN)
	$(CP) $(EXECUTABLE) $(XCLBIN)

check: exe
	export OMP_NUM_THREADS=$(THREAD_NUMBER) && ./$(EXECUTABLE) $(CMD_ARGS)

# Cleaning stuff
RMDIR = rm -rf

clean:
	-$(RMDIR) $(EXECUTABLE)
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) host_src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) _x.* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin


ECHO := @echo
