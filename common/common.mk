.DEFAULT_GOAL := all

.PHONY: all
.PHONY: clean
.PHONY: test

XCRUN     = xcrun
CC        = clang++
LD        = clang++
CD        = cd
RMR       = rm -fr
RM        = rm -f
DIR_GUARD = @mkdir -p $(@D)
PYTHON    = python

METAL_DIR             = metal
METAL_COMMON_DIR      = ../common/metal

OBJ_DIR            = objs
BIN_DIR            = bin
DOC_DIR            = doc

CC_INC             = $(LOCAL_INC) -I$(METAL_DIR) -I../common -I../common/metal
XCRUNCOMPILEFLAGS  = -sdk macosx metal -std=macos-metal2.3 -Werror
XCRUNLINKFLAGS     = -sdk macosx metallib
FRAMEWORKS         = -framework Metal -framework CoreGraphics -framework Accelerate -framework Foundation
CCFLAGS            = -Wall -std=c++20 -stdlib=libc++ -O3 -DTARGET_OS_OSX $(LOCAL_CCFLAGS)
LDFLAGS            = -L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk

METAL_SHADER_BYTECODE = $(patsubst %,$(OBJ_DIR)/%,$(subst .metal,.air,$(subst $(METAL_DIR)/,,$(wildcard $(METAL_DIR)/*.metal))))

OBJS = $(patsubst %,$(OBJ_DIR)/%,$(subst .cpp,.o,$(wildcard *.cpp)))
OBJS += $(patsubst %,$(OBJ_DIR)/%,$(subst .mm,.o,$(wildcard *.mm)))
OBJS += $(patsubst %,$(OBJ_DIR)/%,$(subst .mm,.o,$(wildcard $(METAL_DIR)/*.mm)))
OBJS += $(patsubst %,$(OBJ_DIR)/%,$(subst .cpp,.o,$(wildcard $(METAL_DIR)/*.cpp)))
OBJS += $(patsubst $(METAL_COMMON_DIR)/%,$(OBJ_DIR)/%,$(subst .mm,.o,$(wildcard $(METAL_COMMON_DIR)/*.mm)))
OBJS += $(patsubst %,$(OBJ_DIR)/%,$(subst .mm,.o,$(wildcard $(LOCAL_OBJ_DIR)/*.mm)))
OBJS += $(patsubst %,$(OBJ_DIR)/%,$(subst .cpp,.o,$(wildcard $(LOCAL_OBJ_DIR)/*.cpp)))

METAL_SHADER_LIB = $(BIN_DIR)/$(METAL_SHADER_LIB_NAME).metallib
MAIN_BIN         = $(BIN_DIR)/$(basename $(MAIN_SRC))
MAIN_LOG         = $(DOC_DIR)/make_log.txt
PLOT_SPEC        = $(DOC_DIR)/plot_spec.json
PLOTTER          = ../common/process_log.py
$(OBJ_DIR)/%.air: $(METAL_DIR)/%.metal
	$(DIR_GUARD)
	$(XCRUN) $(XCRUNCOMPILEFLAGS) -c $< -o $@

$(METAL_SHADER_LIB): $(METAL_SHADER_BYTECODE)
	$(DIR_GUARD)
	$(XCRUN) $(XCRUNLINKFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: %.mm
	$(DIR_GUARD)
	$(CC) $(CCFLAGS) $(CC_INC) -c $< -o $@

$(OBJ_DIR)/%.o: $(METAL_COMMON_DIR)/%.mm
	$(DIR_GUARD)
	$(CC) $(CCFLAGS) $(CC_INC) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp
	$(DIR_GUARD)
	$(CC) $(CCFLAGS) $(CC_INC) -c $< -o $@

$(MAIN_BIN): $(OBJS)
	$(DIR_GUARD)
	$(LD) $(LDFLAGS) $(LOCAL_LIBS) $(LOCAL_FRAMEWORKS) $(FRAMEWORKS) $^ -o $@

$(MAIN_LOG): $(MAIN_BIN) $(METAL_SHADER_LIB)
	$(CD) $(BIN_DIR); $(subst $(BIN_DIR)/,./,$(MAIN_BIN)) > ../$@
	$(PYTHON) $(PLOTTER) -logfile $@ -specfile $(PLOT_SPEC) -show_impl -plot_charts -base_dir doc/

all: $(MAIN_LOG)


clean:
	-$(RMR) $(OBJ_DIR) $(BIN_DIR)

