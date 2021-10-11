PRG = mMultGPU
EXECUTABLE = $(PRG).exe

IDIR =./include
NVCC = nvcc

CUDAFLAGS = -I$(IDIR) -O3 -arch=sm_30

OBJECTDIR=./src/obj/

LDIR =./lib
SOURCES = ./src
LIBS=-lcublas 

_DEPS = myFunctions.h timer.h definitions.h typedefs.h params.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJECTS = main.o myFunctions.o 
OBJECTS = $(patsubst %,$(OBJECTDIR)/%,$(_OBJECTS))


$(OBJECTDIR)/%.o: $(SOURCES)/%.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CUDAFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(OBJECTDIR)/*.o *~ $(EXECUTABLE) 

.PHONY:tar

tar:
	tar -cvf $(PRG).tar $(SOURCES) $(DEPS) Makefile
	gzip $(PRG).tar 
