lsfilter ./examples/0_in ./examples/0_in .off | parallel -P $NPROC --timeout $TIMEOUT \
	     meshlabserver -i ./examples/bathtub.obj -o ./examples/bathtub.off;
