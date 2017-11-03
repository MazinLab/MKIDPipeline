/************************************************************************************************ 
 * Bin2HDF.c - A program to convert a sequence of .bin files from the Gen2 readout into a
 *  h5 file.
 *
 * compiled with this command
 /usr/local/hdf5/bin/h5cc -shlib -pthread -o Bin2HDF Bin2HDF.c
 *************************************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <stdint.h>
#include <sys/time.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "hdf5.h"
#include "hdf5_hl.h"

//max number of characters in all strings
#define STR_SIZE 80 

//number of dimensions in the Variable Length array (VLarray).  
//There is a 1D array of pointers to variable length arrays, so rank=1
#define DATA_RANK 1
#define NFIELD 4

// MKID array stats
#define NROACHES 10
#define NPIXELS_PER_ROACH 1024
#define BEAM_ROWS 125
#define BEAM_COLS 80
#define RAD2DEG 57.2957795131

// useful globals
uint32_t residarr[10000] = {0};
uint64_t tstart = 0;

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

typedef struct photon {
    uint32_t timestamp;
    float wvl;
    float wSpec;
    float wNoise;
} photon;

int ParseConfig(int argc, char *argv[], char *Path, int *FirstFile, int *nFiles, char *BeamFile, int *mapflag)
{
    FILE *fp;
    
    fp = fopen(argv[1],"r");
    fscanf(fp,"%s\n",Path);
    fscanf(fp,"%d\n",FirstFile);
    fscanf(fp,"%d\n",nFiles);
    fscanf(fp,"%s\n",BeamFile);
    fscanf(fp,"%d",mapflag);
    fclose(fp);
    
    return 1;
} 

void ParsePacket( uint16_t image[BEAM_COLS][BEAM_ROWS], char *packet, uint64_t l, uint64_t frame[NROACHES])
{
    uint64_t i;
    struct datapacket *data;
    uint64_t swp,swp1;

    for(i=1;i<l/8;i++) {       
       swp = *((uint64_t *) (&packet[i*8]));
       swp1 = __bswap_64(swp);
       data = (struct datapacket *) (&swp1);
       if( data->xcoord >= BEAM_COLS || data->ycoord >= BEAM_ROWS ) continue;
       image[data->xcoord][data->ycoord]++;   
    }
}

void AddPacket(char *packet, uint64_t l, hid_t file_id, size_t dst_size, size_t dst_offset[NFIELD], size_t dst_sizes[NFIELD], int FirstFile, uint32_t BeamMap[BEAM_COLS][BEAM_ROWS], uint64_t *nPhot, uint32_t BeamFlag[BEAM_COLS][BEAM_ROWS], int mapflag, char ResIdString[BEAM_COLS][BEAM_ROWS][20], photon *ptable[BEAM_COLS][BEAM_ROWS], uint32_t ptablect[BEAM_COLS][BEAM_ROWS] )
{
    uint64_t i,swp,swp1,swp2,swp3;
    int64_t basetime;
    uint32_t rid,roach;
    struct hdrpacket *hdr;
    struct datapacket *data;
    photon p;
    
    // get info form header packet
    swp = *((uint64_t *) (&packet[0]));
    swp1 = __bswap_64(swp);
    hdr = (struct hdrpacket *) (&swp1);             
    if (hdr->start != 0b11111111) {
        printf("Error - packet does not start with a correctly formatted header packet!\n");
        return;
    }
    
    // if no start timestamp, store start timestamp
    if( tstart == 0 ) {
		tstart = (uint64_t) hdr->timestamp;
		//printf("Start time = %ld from ROACH %d\n",tstart,hdr->roach); fflush(stdout);
	}
    basetime = hdr->timestamp - tstart; // time since start of first file
    
    if( basetime < 0 ) { // maybe have some packets out of order early in file		
	    printf("Early Start!\n");
		basetime = 0; 
	}
    
    for(i=1;i<l/8;i++) {
       
		swp = *((uint64_t *) (&packet[i*8]));
		swp1 = __bswap_64(swp);
		data = (struct datapacket *) (&swp1);
		if( data->xcoord >= BEAM_COLS || data->ycoord >= BEAM_ROWS ) continue;
		if( mapflag > 0 && BeamFlag[data->xcoord][data->ycoord] > 0) continue ; // if mapflag is set only record photons that were succesfully beammapped       

		// skip if we have some issue that takes us over 2500 cts/sec
		if( ptablect[data->xcoord][data->ycoord] > 2498 ) continue;

		// add the photon to ptable and increment the appropriate counter
		ptable[data->xcoord][data->ycoord][ptablect[data->xcoord][data->ycoord]].timestamp = (uint32_t) (basetime*500 + data->timestamp);
		ptable[data->xcoord][data->ycoord][ptablect[data->xcoord][data->ycoord]].wvl = ((float) data->wvl)*RAD2DEG/32768.0;
		ptable[data->xcoord][data->ycoord][ptablect[data->xcoord][data->ycoord]].wSpec = 1.0;
		ptable[data->xcoord][data->ycoord][ptablect[data->xcoord][data->ycoord]].wNoise = 1.0;
		ptablect[data->xcoord][data->ycoord]++;

    }
}

void ParseBeamMapFile(char *BeamFile, uint32_t BeamMap[BEAM_COLS][BEAM_ROWS], uint32_t BeamFlag[BEAM_COLS][BEAM_ROWS])
{
    // read in Beam Map file
    // format is [ResID, flag, X, Y], X is [0, 79], Y is [0, 124]
    
    FILE *fp;
    int ret, resid, flag, x, y;

    fp = fopen(BeamFile, "r");    
    do {
        ret = fscanf(fp,"%d %d %d %d\n", &resid, &flag, &x, &y);
        //printf("%d %d %d %d\n", resid, flag, x, y);
        BeamMap[x][y] = resid;
        BeamFlag[x][y] = flag;      
    } while ( ret == 4);    
    fclose(fp);
}

int main(int argc, char *argv[])
{
    char path[STR_SIZE], fName[STR_SIZE], BeamFile[STR_SIZE], outfile[STR_SIZE], imname[STR_SIZE], tname[STR_SIZE];
    int FirstFile, nFiles,mapflag;
    long fSize, rd, j, k;
    struct stat st;
    FILE *fp;
    uint64_t **data, *dSize;
    clock_t start, diff, olddiff;
    uint64_t swp,swp1,i,pstart,pcount,tStart,firstHeader, nPhot, tPhot=0;
    struct hdrpacket *hdr;
    char packet[808*16];
    char *olddata;
    uint16_t image[BEAM_COLS][BEAM_ROWS];
    unsigned char smimage[BEAM_COLS*BEAM_ROWS];
    uint64_t frame[NROACHES];
    uint32_t BeamMap[BEAM_COLS][BEAM_ROWS] = {0};
    uint32_t BeamFlag[BEAM_COLS][BEAM_ROWS] = {0};
    char ResIdString[BEAM_COLS][BEAM_ROWS][20];
    char addHeaderCmd[] = "python addH5Header.py ";
    photon p1;
    
    photon *ptable[BEAM_COLS][BEAM_ROWS];
    uint32_t ptablect[BEAM_COLS][BEAM_ROWS] = {0};
    
    // hdf5 variables
    hid_t file_id;
    hid_t gid_beammap, sid_beammap, did_beammap, did_flagmap, gid_photons, sid_photons, did_photons;
    herr_t status;
    hsize_t dims[2];
    
    size_t dst_size =  sizeof(photon);
    size_t dst_offset[NFIELD] = { HOFFSET( photon, timestamp ), HOFFSET( photon, wvl ), HOFFSET( photon, wSpec ), HOFFSET( photon, wNoise) };
    size_t dst_sizes[NFIELD] = { sizeof(p1.timestamp), sizeof(p1.wvl),sizeof(p1.wSpec), sizeof(p1.wNoise) };

    /* Define field information */
    const char *field_names[NFIELD]  =
    { "Time","Wavelength","Spec Weight","Noise Weight"};
    hid_t      field_type[NFIELD];
    hid_t      string_type;
    hsize_t    chunk_size = 10;
    int        *fill_data = NULL;
    int        compress  = 0;

    /* Initialize field_type */
    field_type[0] = H5T_STD_U32LE;
    field_type[1] = H5T_NATIVE_FLOAT;
    field_type[2] = H5T_NATIVE_FLOAT;
    field_type[3] = H5T_NATIVE_FLOAT;      
        
    memset(packet, 0, sizeof(packet[0]) * 808 * 16);    // zero out array
    
    // Open config file and parse    
    if( argc != 2 ) {
        printf("Bin2HDF error - First command line argument must be the configuration file.\n");
        exit(0);
    }
    if (ParseConfig(argc,argv,path,&FirstFile,&nFiles,BeamFile,&mapflag) == 0 ) {
        printf("Bin2HDF error - Config parsing error.\n");
		exit(1);
	}
    
    // Set up memory structure for data
    data = (uint64_t **) malloc( nFiles * sizeof(uint64_t *) );
    dSize = (uint64_t *) malloc( nFiles * sizeof(uint64_t *) );
        
    // Read in beam map and parse it make 2D beam map and flag arrays
    ParseBeamMapFile(BeamFile,BeamMap,BeamFlag);
    printf("Parsed beam map.\n"); fflush(stdout);
    
    // Read all the data into memory
    start = clock();
    for(i=0; i < nFiles; i++) {
        sprintf(fName,"%s/%ld.bin",path,FirstFile+i);
        stat(fName, &st);
        fSize = st.st_size;
        printf("Reading %s - %ld bytes\n",fName,fSize);
        data[i] = (uint64_t *) malloc( fSize * sizeof(uint64_t *) );
        dSize[i] = (uint64_t) fSize;
        
        fp = fopen(fName, "rb");
        rd = fread( data[i], 1, fSize, fp);
        if( rd != fSize) printf("Didn't read the entire file %s\n",fName);
        fclose(fp);        
    }
    diff = clock()-start;
    olddiff = diff;
    printf("Read data to memory in %f ms.\n",(float)diff*1000.0/CLOCKS_PER_SEC);
    
    // Create H5 file and set attributes
    sprintf(outfile,"%s/%d.h5",path,FirstFile);
    file_id = H5Fcreate (outfile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    // put the beam map into the h5 file    
    dims[0] = BEAM_COLS;
    dims[1] = BEAM_ROWS;
    gid_beammap = H5Gcreate2(file_id, "BeamMap", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    sid_beammap = H5Screate_simple(2, dims, NULL);
    did_beammap = H5Dcreate2(file_id, "/BeamMap/Map", H5T_NATIVE_UINT, sid_beammap, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    did_flagmap = H5Dcreate2(file_id, "/BeamMap/Flag", H5T_NATIVE_UINT, sid_beammap, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    H5Dwrite (did_beammap, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, BeamMap);
    H5Dwrite (did_flagmap, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, BeamFlag);
        
    H5Dclose(did_beammap);
    H5Dclose(did_flagmap);
    H5Sclose(sid_beammap);
    H5Gclose(gid_beammap);   
    
    gid_beammap = H5Gcreate2(file_id, "Images", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // Step through .bin files that are now in memory, suck out the data, convert it into the new packet format 
    // and write it to the h5 file
    start = clock();
  
    gid_photons = H5Gcreate2(file_id, "Photons", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
    // make photon tables for every resid
    for(i=0; i < BEAM_COLS; i++) {
		for(j=0; j < BEAM_ROWS; j++) {
			if( BeamMap[i][j] == 0 ) continue;
		
			ptable[i][j] = (photon *) malloc( 2500 * sizeof(photon) );	// allocate memory for ptable 
			sprintf(tname,"/Photons/%d",BeamMap[i][j]);			
			memcpy(ResIdString[i][j],tname,20); 	// store the table name string in an array
			// make the table
			H5TBmake_table( "Photon Data", file_id, ResIdString[i][j], NFIELD, 0, dst_size, field_names, dst_offset, field_type,chunk_size, fill_data, compress, &p1);
		}
	}
	
	printf("Made individual photon data tables.\n"); fflush(stdout);
        
    for(i=0; i < nFiles; i++) {
        olddata = (char *) data[i];
        pstart = 0;
        pcount = 0;
        nPhot = 0;
        memset(ptablect,0,sizeof(uint32_t)*BEAM_COLS*BEAM_ROWS); // zero out the count table
        
        printf("File %ld: ",i);    
            
        // .bin may not always start with a header packet, so search until we find the first header
        for( j=0; j<dSize[i]/8; j++) {
            swp = *((uint64_t *) (&olddata[j*8]));
            swp1 = __bswap_64(swp);
            hdr = (struct hdrpacket *) (&swp1);
            if (hdr->start == 0b11111111) {
                firstHeader = j;
                pstart = j;
                if( firstHeader != 0 ) printf("First header at %ld\n",firstHeader);
                break;     
            }      
        }
                
        // reformat all the packets into memory then dump to disk for speed
        for( j=firstHeader+1; j<dSize[i]/8; j++) {
 
            swp = *((uint64_t *) (&olddata[j*8]));
            swp1 = __bswap_64(swp);
            hdr = (struct hdrpacket *) (&swp1);             
                                      
            if (hdr->start == 0b11111111) {        // found new packet header!
                // fill packet and parse
                //printf("Found next header at %d\n",j*8); fflush(stdout);
                memmove(packet,&olddata[pstart],j*8 - pstart);
                pcount++;
                // parse into image                
                ParsePacket(image,packet,j*8 - pstart,frame); 
                // add to HDF5 file
                AddPacket(packet,j*8-pstart,file_id,dst_size,dst_offset,dst_sizes,FirstFile,BeamMap,&nPhot,BeamFlag,mapflag,ResIdString,ptable,ptablect);
		        pstart = j*8;   // move start location for next packet
		        if( pcount%1000 == 0 ) printf("."); fflush(stdout);	                      
            }
        }
        
        // save photon tables to hdf5
        for(j=0; j < BEAM_COLS; j++) {
			for(k=0; k < BEAM_ROWS; k++) {
				if( BeamMap[j][k] == 0 ) continue;
				if( ptablect[j][k] == 0 ) continue;    
				//printf("%s %ld\n", ResIdString[j][k], ptablect[j][k]);
				//printf("%d %d %s %ld\n", j, k, ResIdString[j][k], ptablect[j][k]); fflush(stdout);
				H5TBappend_records(file_id, ResIdString[j][k], ptablect[j][k], dst_size, dst_offset, dst_sizes, ptable[j][k] );
				nPhot +=  ptablect[j][k];  
			}
		}
		
		printf("|"); fflush(stdout);
        
        // save image array to hdf5
        for( j=0; j < BEAM_COLS*BEAM_ROWS; j++ ) {
            smimage[j] =  (unsigned char) (image[j%BEAM_COLS][j/BEAM_COLS]/10);
            if( smimage[j] > 2499 ) smimage[j] = 0;
        }
                
        sprintf(imname,"/Images/%ld",i+FirstFile);
        H5IMmake_image_8bit( file_id, imname, (hsize_t)BEAM_COLS, (hsize_t)BEAM_ROWS, smimage );
        memset(image, 0, BEAM_COLS*BEAM_ROWS*sizeof(uint16_t));
        
        printf(" %ld packets, %ld photons. %ld photons/packet.\n",pcount,nPhot,nPhot/pcount);
        tPhot += nPhot;
    }

    H5Gclose(gid_photons);   
    
    diff = clock()-start;
    printf("Parsed %ld photons in %f seconds: %9.1f photons/sec.\n",tPhot,(float)(diff+olddiff)/CLOCKS_PER_SEC,((float)tPhot)/((float)(diff+olddiff)/CLOCKS_PER_SEC));

    // Close up
    H5Gclose(gid_beammap);   
    H5Fclose(file_id);
 
     // make photon tables for every resid
    for(i=0; i < BEAM_COLS; i++) {
		for(j=0; j < BEAM_ROWS; j++) {
			if( BeamMap[i][j] == 0 ) continue;
			free(ptable[i][j]);
		}
	}
    
    for(i=0; i < nFiles; i++) free(data[i]); 
    free(data);
    free(dSize);

    strcat(addHeaderCmd, argv[1]);
    strcat(addHeaderCmd, " ");
    strcat(addHeaderCmd, outfile);
    system(addHeaderCmd);

}
