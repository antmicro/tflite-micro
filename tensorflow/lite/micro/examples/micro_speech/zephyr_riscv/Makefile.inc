ifeq ($(TARGET),zephyr_vexriscv)

export ZEPHYR_TOOLCHAIN_VARIANT?=zephyr
export TOOLCHAIN_BASE=${ZEPHYR_SDK_INSTALL_DIR}/riscv64-zephyr-elf/riscv64-zephyr-elf
export TOOLCHAIN_VERSION=9.2.0

PROJECT_INCLUDES += \
	. \
	${CURDIR} \
	${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION} \
	${TOOLCHAIN_BASE}/include \
	${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION}/riscv64-zephyr-elf/rv32i/ilp32
MICROLITE_LIBS += $(LINKER_FLAGS)
MICRO_SPEECH_HDRS += tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h

ZEPHYR_MICRO_SPEECH_SRCS = \
	tensorflow/lite/micro/examples/micro_speech/main.cc \
	tensorflow/lite/micro/examples/micro_speech/main_functions.cc \
	tensorflow/lite/micro/examples/micro_speech/zephyr_riscv/audio_provider.cc \
	tensorflow/lite/micro/examples/micro_speech/feature_provider.cc \
	tensorflow/lite/micro/examples/micro_speech/recognize_commands.cc \
	tensorflow/lite/micro/examples/micro_speech/command_responder.cc \
	tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
	tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc \
	tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
	tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc \
	$(MICRO_FEATURES_GENERATOR_SRCS) \
	boards/litex_vexriscv.overlay \
	prj.conf \
	$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/micro_speech/micro_speech_model_data.cc

$(eval $(call generate_project,cmake,zephyr_cmake_project.cmake,micro_speech,$(MICROLITE_CC_SRCS) $(MICROLITE_CC_KERNEL_SRCS) $(THIRD_PARTY_CC_SRCS) $(ZEPHYR_MICRO_SPEECH_SRCS) $(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(MICRO_SPEECH_HDRS),,$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS),$(CCFLAGS),))

$(PRJDIR)micro_speech/cmake/CMakeLists.txt: $(PRJDIR)micro_speech/cmake/zephyr_cmake_project.cmake
	@sed -E 's#\%\{INCLUDE_DIRS\}\%#$(PROJECT_INCLUDES)#g' $< > $@

micro_speech_pregen_arrays:
	@python3 tensorflow/lite/micro/tools/generate_cc_arrays.py ${PRJDIR}micro_speech/cmake/ ${MICRO_SPEECH_GENERATOR_INPUTS}

# We are skipping here copy of `zephyr` third_party repository
# To compile standalone project ZEPHYR_BASE enviroment variable should be set
micro_speech_generate_project: generate_micro_speech_cmake_project $(PRJDIR)micro_speech/cmake/CMakeLists.txt
	( \
		. ${ZEPHYR_BASE}/venv-zephyr/bin/activate; \
		cmake -B${GENDIR}micro_speech/build -DBOARD="litex_vexriscv" -H${PRJDIR}micro_speech/cmake/ -DPython_ROOT_DIR=${ZEPHYR_BASE}/venv-zephyr/bin/; \
		make -C ${GENDIR}micro_speech/build; \
	)

micro_speech_bin: micro_speech_pregen_arrays micro_speech_generate_project

endif
