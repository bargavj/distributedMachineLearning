#pragma once

void ocTestUtilTcpOrDie(struct ProtocolDesc* pd,const char* remote_host,
                        const char* port);
double wallClock();

int hex_to_int(char c);

int hex_to_ascii(char c, char d);

void hex_string_to_char_array(const char * hex_string);

bool compare(uint8_t *x, uint8_t *y, int len);
