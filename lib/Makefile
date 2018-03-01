SUFFIXES += .d

CXX_SOURCES   := $(shell find src -name "*.cpp")
CXX_DEPFILES  := $(patsubst src/%.cpp,deps/%.d, $(CXX_SOURCES))
CXX_OBJECTS   := $(patsubst src/%.cpp,obj/%.o, $(CXX_SOURCES))
EXECUTABLES   := bin/test_perf_nvme bin/test_perf_memcpy
LIBS          := bin/libnvme_sampler.a bin/libnvme_sampler.so

NODEPS := clean

HOST_COMPILER        ?= g++-7
CXXFLAGS_CPU         ?= -mfma -mavx2 -ggdb -mavx2 -msse4.1 -msse -O3 -fno-omit-frame-pointer
CXXFLAGS             := -m64 -std=c++1z -Wall -Wextra -I ~/torch/install/include $(CXXFLAGS_CPU)

all: build

build: deps obj bin $(EXECUTABLES) $(LIBS)

bin:
	@mkdir bin

obj:
	@mkdir obj

deps:
	@mkdir deps

clean:
	rm -rf src/*.d bin obj src/*.o deps
 
$(CXX_DEPFILES): deps/%.d: src/%.cpp deps
	@echo cxx_dep $@
	@$(HOST_COMPILER) -MM -MT '$(patsubst src/%.cpp,obj/%.o,$<)' $< -MF $@

$(CXX_OBJECTS): obj/%.o: src/%.cpp deps/%.d
	@echo cxx $@
	@$(HOST_COMPILER) ${CXXFLAGS} -o $@ -c $<

$(EXECUTABLES): bin/% : obj/%.o
	@echo cxx_link $@
	@$(HOST_COMPILER) ${CXXFLAGS} -o $@ $+ -laio -pthread

bin/libnvme_sampler.a: obj/nvme_api.o
	@echo ar  $@
	@ar rcs -o $@ $+

bin/libnvme_sampler.so: obj/nvme_api.o
	@echo link  $@
	@$(HOST_COMPILER) ${CXXFLAGS} -o $@ src/nvme_api.cpp -laio -pthread -fPIC -shared


ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS)))) # ignore dep. generation when cleaning
-include $(CXX_DEPFILES)
endif
