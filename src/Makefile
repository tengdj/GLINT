CC = gcc
CXX = g++
NVCC = nvcc

SRCS := $(wildcard */*.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

INCLUDES	= -I /usr/include -I ./src
LIBS		= -L/usr/lib/x86_64-linux-gnu -lboost_program_options -lpthread
CPPFLAGS	= -g -std=c++14 -Wno-cpp

CUDALIBS	= -L/usr/local/cuda/lib64 -lcuda -lcudart

DATAGEN_SRCS := $(wildcard datagen/*.cpp)
DATAGEN_OBJS := $(patsubst %.cpp,%.o,$(DATAGEN_SRCS))


#compile all the cu files
CU_SRCS := $(wildcard cuda/*.cu)
CU_OBJS := $(patsubst %.cu,%_cu.o,$(CU_SRCS))

#%_cu.o: %.cu
#	$(NVCC) -c $(CPPFLAGS) $(INCLUDES) -o $@ $<	
	
%.o:	%.cpp
	$(CXX) -c $(CPPFLAGS) $(INCLUDES) -o $@ $<	

all:	datagen


# for macro queries

datagen:	datagen/datagen.o $(DATAGEN_OBJS) 
	$(CXX) -o ../build/$@ $^ $(LIBS) 
	

clean:
	rm -fr ../build/* $(OBJS) $(CU_OBJS)
