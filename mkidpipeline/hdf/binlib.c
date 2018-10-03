/*
binlib.c
Written by Ben 9/2018
Mostly chopped from the more involved bin2hdf

The code is basic. It reads a bin file, turns the bits in the packet into arrays based on their location in the packet.

*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

struct datapacket {
    int baseline:17;
    int wvl:18;
    unsigned int timestamp:9;
    unsigned int ycoord:10;
    unsigned int xcoord:10;
}__attribute__((packed));;

struct hdrpacket {
    unsigned long timestamp:36;
    unsigned int frame:12;
    unsigned int roach:8;
    unsigned int start:8;
}__attribute__((packed));;

long cparsebin(const char *fName, unsigned long max_len,
                       int* baseline, int* wavelength, unsigned long* timestamp,
                       unsigned int* ycoord, unsigned int* xcoord, unsigned int* roach) {
    /*
    The function returns the number of packet in the file. If the file turns out to have more packets than max_len,
    the arrays are populated with the first max_len-1 records and the last record.
    If there are errors (e.g. file not found) return appropriate error numbers as - return values.
    */
    unsigned long out_i=0, pcount=0;
	FILE *fp;
	struct stat st;
	long fSize,rd;
	uint64_t *data;
	uint64_t swp,swp1,firstHeader,pstart,curtime=0,curroach=0;
	struct hdrpacket *hdr;
	struct datapacket *photondata;
	char packet[808*16];

    //open up file
	stat(fName, &st);
	fSize = st.st_size;
	printf("\nReading %s - %ld bytes\n",fName,fSize);
	data = (uint64_t *) malloc(fSize);
    fp = fopen(fName, "rb");
    rd = fread( data, 1, fSize, fp);
    if( rd != fSize) printf("Didn't read the entire file %s\n",fName);
    fclose(fp);

    //if not open
    if (rd < 0 ) return -1;

	// Find the first header packet
	for(unsigned long i=0; i<fSize/8; i++) {
		swp = *((uint64_t *) (&data[i]));
		swp1 = __bswap_64(swp);
		hdr = (struct hdrpacket *) (&swp1);
		if (hdr->start == 0b11111111) {
			firstHeader = i;
			pstart = i;
			curtime = hdr->timestamp;
			curroach = hdr->roach;
			if( firstHeader != 0 ) printf("First header at %ld\n",firstHeader);
			break;
		}
	}

	// New approach - do it all in this function
    for(unsigned long i=firstHeader+1; i<fSize/8; i++) {
        swp = *((uint64_t *) (&data[i]));
        swp1 = __bswap_64(swp);
        hdr = (struct hdrpacket *) (&swp1);
        if (hdr->start == 0b11111111) {        // found new packet header - update timestamp and curroach
			curtime = hdr->timestamp;
			curroach = hdr->roach;
		}
		else {                              // must be data. Save as photondata struct
		    out_i = pcount >= max_len ? max_len: pcount;
			photondata = (struct datapacket *) (&swp1);
			baseline[out_i] = photondata->baseline;
			wavelength[out_i] = photondata->wvl;
			timestamp[out_i] = photondata->timestamp + curtime;
			ycoord[out_i] = photondata->ycoord;
			xcoord[out_i] = photondata->xcoord;
			roach[out_i] = curroach;
			pcount++;
		}

	}

    //close up file
	free(data);

    return pcount;
}
