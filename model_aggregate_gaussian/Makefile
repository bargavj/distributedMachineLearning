testName=modelAggregate

CILPATH=~/obliv-c
ACKPATH=~/absentminded-crypto-kit
CODEPATH=~/distributedMachineLearning
REMOTE_HOST=localhost
CFLAGS=-DREMOTE_HOST=$(REMOTE_HOST) -O3
./a.out: $(testName).oc $(testName).c $(CODEPATH)/util.c $(ACKPATH)/build/lib/liback.a $(CILPATH)/_build/libobliv.a 
	$(CILPATH)/bin/oblivcc $(CFLAGS) -I . -I $(ACKPATH)/src/ $(testName).oc $(testName).c $(CODEPATH)/util.c $(ACKPATH)/src/obig.oc $(CODEPATH)/ofixed.oc -lm
clean:
	rm -f a.out
