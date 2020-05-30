#include <stdio.h>
#include <stdlib.h>
#include <obliv.h>
#include <unistd.h>
#include "modelAggregate.oh"

int main(int argc, char *argv[]) {
  ProtocolDesc pd;     // Protocol Descriptor used for passing message between the parties
  protocolIO io;       // Data structure to store the input parameters
  double start, end;   // Variables to measure start and end time of Wall Clock
  int party;           // CommandLine Argument: 1 -> Generator and 2 -> Evaluator
  FILE *fptr1, *fptr2, *fptr3, *fptr4, *fptr5, *fptr6, *fptr7, *fptr8, *fptr9;
  int i, j, bytes;
  char c[20];
  int M, D;

  if (strcmp(argv[2], "--") == 0) {
    party = 1;
  } else {
    party = 2;
  }

  M = atoi(argv[4]);
  D = atoi(argv[5]);
  io.M = M;
  io.D = D; 
  io.beta1 = (int64_t **)malloc(M * sizeof(int64_t *));
  io.random11 = (int64_t *)malloc(D * sizeof(int64_t));
  io.random21 = (int64_t *)malloc(D * sizeof(int64_t));
  io.beta2 = (int64_t **)malloc(M * sizeof(int64_t *));
  io.random12 = (int64_t *)malloc(D * sizeof(int64_t));
  io.random22 = (int64_t *)malloc(D * sizeof(int64_t));
  for(i = 0; i < M; i++) {
    io.beta1[i] = (int64_t *)malloc(D * sizeof(int64_t));
    io.beta2[i] = (int64_t *)malloc(D * sizeof(int64_t));
  }
  io.beta_avg = (int64_t *)malloc(D * sizeof(int64_t));

  if(party == 1) {
    fptr1=fopen("Inputs/beta1.txt","r");
    for(i = 0; i < M; i++) {
      for(j = 0; j < D; j++) {
        bytes = fscanf(fptr1,"%s", c);
        io.beta1[i][j] = atoll(c);
      }
    }
    fclose(fptr1);
    fptr2=fopen("Inputs/noise1.txt","r");
    bytes = fscanf(fptr2,"%s", c);
    io.noise1 = atoll(c);
    fclose(fptr2);
    fptr3=fopen("Inputs/random11.txt","r");
    for(i = 0; i < D; i++) {
      bytes = fscanf(fptr3,"%s", c);
      io.random11[i] = atoll(c);
    }
    fclose(fptr3);
    fptr4=fopen("Inputs/random21.txt","r");
    for(i = 0; i < D; i++) {
      bytes = fscanf(fptr4,"%s", c);
      io.random21[i] = atoll(c);
    }
    fclose(fptr4);
  }

  if(party == 2) {
    fptr5=fopen("Inputs/beta2.txt","r");
    for(i = 0; i < M; i++) {
      for(j = 0; j < D; j++) {
        bytes = fscanf(fptr5,"%s", c);
        io.beta2[i][j] = atoll(c);
      }
    }
    fclose(fptr5);
    fptr6=fopen("Inputs/noise2.txt","r");
    bytes = fscanf(fptr6,"%s", c);
    io.noise2 = atoll(c);
    fclose(fptr6);
    fptr7=fopen("Inputs/random12.txt","r");
    for(i = 0; i < D; i++) {
      bytes = fscanf(fptr7,"%s", c);
      io.random12[i] = atoll(c);
    }
    fclose(fptr7);
    fptr8=fopen("Inputs/random22.txt","r");
    for(i = 0; i < D; i++) {
      bytes = fscanf(fptr8,"%s", c);
      io.random22[i] = atoll(c);
    }
    fclose(fptr8);
    sleep(1);
  }

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
    fptr9=fopen("Output/beta_avg.txt","w");
    for(i = 0; i < D; i++) 
      fprintf(fptr9, "%.8f ", io.beta_avg[i]*1.0/SCALE);
    fclose(fptr9);
  }

  for(i = 0; i < M; i++) {
    free(io.beta1[i]);
    free(io.beta2[i]);
  }
  free(io.beta1);
  free(io.random11);
  free(io.random21);
  free(io.beta_avg);
  free(io.beta2);
  free(io.random12);
  free(io.random22);

  cleanupProtocol(&pd);

  if(io.correct == 0) {
    return 1;
  }

  return 0;
}
