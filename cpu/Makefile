PRG = mMultCPU
EXECUTABLE = $(PRG).exe
IDIR =./include
CC=gcc
DEBUG = -g 
CFLAGS=-Wall -O2 -std=c99 $(DEBUG) -I$(IDIR) 

OBJECTDIR=./src/obj/

LDIR =./lib
SOURCES = ./src
LIBS=-lcblas 

_DEPS = myFunctions.h timer.h definitions.h typedefs.h params.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJECTS = main.o myFunctions.o 
OBJECTS = $(patsubst %,$(OBJECTDIR)/%,$(_OBJECTS))


$(OBJECTDIR)/%.o: $(SOURCES)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(OBJECTDIR)/*.o *~ $(EXECUTABLE) 

.PHONY:tar

tar:
	tar -cvf $(PRG).tar $(SOURCES) $(DEPS) Makefile
	gzip $(PRG).tar 
