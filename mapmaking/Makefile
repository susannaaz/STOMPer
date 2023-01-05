#CCC = mpic++ -Wall -Wextra -g -O0
CCC = mpic++ -g -O0
CC = g++ -g -O0

FITS = -I/group/cmb/litebird/common_tools/custom_install/cfitsio-4.1.0/ -L/group/cmb/litebird/common_tools/custom_install/cfitsio-4.1.0/ -lcfitsio 
FFTW = -I/sw/packages/fftw/fftw3/include/ -L/sw/packages/fftw/fftw3/lib/ -lfftw3

mpi : Sanepic.cc
	${CCC} $(FITS) $(FFTW) Sanepic.cc -o Sanepic

mpi_test : Sanepic.cc
	${CC} $(FITS) $(FFTW) Sanepic.cc -o Sanepic

test : Sanepic_test.cc
	${CC} $(FITS) $(FFTW) Sanepic_test.cc -o Sanepic_test 

clean :
	-rm Sanepic *.o core.*

