CC = gcc
CXX = g++
NVCC = nvcc

SRCS := $(wildcard */*.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

INCLUDES	= -I /usr/include -I ./src
LIBS		= -L/usr/lib/x86_64-linux-gnu -lboost_program_options -lpthread
CPPFLAGS	= -g -std=c++14 -Wno-cpp

CUDALIBS	= -L/usr/local/cuda/lib64 -lcuda -lcudart

GEOMETRY_SRCS := $(wildcard geometry/*.cpp)
GEOMETRY_OBJS := $(patsubst %.cpp,%.o,$(GEOMETRY_SRCS))

TRACING_SRCS := $(wildcard tracing/*.cpp)
TRACING_OBJS := $(patsubst %.cpp,%.o,$(TRACING_SRCS))

#compile all the cu files
CU_SRCS := $(wildcard cuda/*.cu)
CU_OBJS := $(patsubst %.cu,%_cu.o,$(CU_SRCS))

%_cu.o: %.cu
	$(NVCC) -c -g -std=c++14 $(INCLUDES) -o $@ $<	
	
%.o:	%.cpp
	$(CXX) -c $(CPPFLAGS) $(INCLUDES) -o $@ $<	

all:	datagen trace


# for macro queries

datagen:	tools/datagen.o $(GEOMETRY_OBJS) $(TRACING_OBJS) 
	$(CXX) -o ../build/$@ $^ $(LIBS) 

cudatrace:	tools/trace.o $(GEOMETRY_OBJS) $(TRACING_OBJS) $(CU_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) $(CUDALIBS)
	
trace:	tools/trace.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 
	
cudatest:	tools/cudatest.o $(CU_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) $(CUDALIBS)
	
clean:
	rm -fr ../build/* $(OBJS) $(CU_OBJS)
