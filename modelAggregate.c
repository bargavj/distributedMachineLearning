#include <stdio.h>
#include <stdlib.h>
#include <obliv.h>
#include "modelAggregate.oh"

int main(int argc, char *argv[]) {
  ProtocolDesc pd;     // Protocol Descriptor used for passing message between the parties
  protocolIO io;       // Data structure to store the input parameters
  double start, end;   // Variables to measure start and end time of Wall Clock
  int party;           // CommandLine Argument: 1 -> Generator and 2 -> Evaluator
  FILE *fptr1, *fptr2, *fptr3, *fptr4;
  int i, j, bytes;
  char c[20];
  
  if (strcmp(argv[2], "--") == 0) {
    party = 1;
  } else {
    party = 2;
  }
  
  if(party == 1) {
    fptr1=fopen("Inputs/beta1.txt","r");
    for(i = 0; i < M; i++) {
      for(j = 0; j < D; j++) {
        bytes = fscanf(fptr1,"%s", c);
        io.beta1[i][j] = atoll(c);
      }
    }
    fclose(fptr1);
  }
  if(party == 2) {
    fptr2=fopen("Inputs/beta2.txt","r");
    for(i = 0; i < M; i++) {
      for(j = 0; j < D; j++) {
        bytes = fscanf(fptr2,"%s", c);
        io.beta2[i][j] = atoll(c);
      }
    }
    fclose(fptr2);
  }
  
  fptr3=fopen("Inputs/random_vals.txt","r");
  for(i = 0; i < M; i++) {
    for(j = 0; j < D; j++) {
      bytes = fscanf(fptr3,"%s", c);
      io.random_vals[i][j] = atoll(c);
    }
  }
  fclose(fptr3);
  
  for(i = 0; i < M; i++) {
    io.sizes[i] = 500;
  }
  
  io.epsilon = 0.5;
  io.lambda = 0.00316227766;//0.00316227766;// 10^-2.5//0.03162277660168379; // 10^-1.5
  
  setCurrentParty(&pd, party);

  const char* remote_host = (strcmp(argv[2], "--") == 0 ? NULL : argv[2]);
  ocTestUtilTcpOrDie(&pd, remote_host, argv[1]);

  if (strcmp(argv[3], "yao") == 0) {
    io.proto = 1;
  } else if (strcmp(argv[3], "dualex") == 0) {
    io.proto = 2;
  } else {
    io.proto = 0;
  }

  start = wallClock();

  if (io.proto == 1) {
    execYaoProtocol(&pd, aggregate, &io);
  } else {
    execDualexProtocol(&pd, aggregate, &io);
  }

  end = wallClock();

  fprintf(stderr, "\nParty %d, Elapsed Time: %f seconds \n", party, end - start);

  if (party == 1) {
    fptr4=fopen("Output/beta_avg.txt","w");
    for(i = 0; i < D; i++) 
      fprintf(fptr4, "%.8f ", io.beta_avg[i]*1.0/SCALE);
    fclose(fptr4);
  }

  cleanupProtocol(&pd);

  return 0;
}
