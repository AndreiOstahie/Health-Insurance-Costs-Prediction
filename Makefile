all: serial predict_on_test predict_on_custom

LDLIBS = -lm
CFLAGS = -pg

serial: serial.o

serial.o: serial.c

predict_on_test: predict_on_test.o

predict_on_test.o: predict_on_test.c

predict_on_custom: predict_on_custom.o

predict_on_custom.o: predict_on_custom.c

clean:
	rm -rf serial predict_on_test predict_on_custom *.o