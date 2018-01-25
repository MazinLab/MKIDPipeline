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
#define NPIXELS_PER_ROACH 1024
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

int ParseConfig(int argc, char *argv[], char *Path, int *FirstFile, int *nFiles, char *BeamFile, int *mapflag, int *beamCols, int *beamRows)
{
    FILE *fp;
    
    fp = fopen(argv[1],"r");
    fscanf(fp,"%d %d\n", beamCols, beamRows);
    fscanf(fp,"%s\n",Path);
    fscanf(fp,"%d\n",FirstFile);
    fscanf(fp,"%d\n",nFiles);
    fscanf(fp,"%s\n",BeamFile);
    fscanf(fp,"%d",mapflag);
    fclose(fp);
    
    return 1;
} 

void ParsePacket( uint16_t **image, char *packet, uint64_t l, uint64_t *frame, int beamCols, int beamRows)
{
    uint64_t i;
    struct datapacket *data;
    uint64_t swp,swp1;

    for(i=1;i<l/8;i++) {       
       swp = *((uint64_t *) (&packet[i*8]));
       swp1 = __bswap_64(swp);
       data = (struct datapacket *) (&swp1);
       if( data->xcoord >= beamCols || data->ycoord >= beamRows ) continue;
       image[data->xcoord][data->ycoord]++;   
    }
}

void AddPacket(char *packet, uint64_t l, hid_t file_id, size_t dst_size, size_t dst_offset[NFIELD], size_t dst_sizes[NFIELD], int FirstFile, uint32_t **BeamMap, uint64_t *nPhot, uint32_t **BeamFlag, int mapflag, char ***ResIdString, photon ***ptable, uint32_t **ptablect, int beamCols, int beamRows )
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
		if( data->xcoord >= beamCols || data->ycoord >= beamRows ) continue;
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

/*
 * Sorts all photon tables in time order. Uses insertion sort (good for mostly ordered data)
 */
void SortPhotonTables(photon ***ptable, uint32_t **ptablect, int beamCols, int beamRows)
{
    photon *photonToSortAddr; //address of element currently being sorted
    photon *curPhotonAddr; //address of element being compared to photonToSort
    photon *photonSwapAddr; //address of element being moved, once correct index for photonToSort has been found
    photon photonToSort; //stores the data in photonToSortAddr
    int x,y; //beammap indices

    for(x=0; x<beamCols; x++)
        for(y=0; y<beamRows; y++)
            //loop through photons in list, check if it is greater than previous elements (all previous elements are already sorted)
            for(photonToSortAddr = ptable[x][y]+1; photonToSortAddr < ptable[x][y] + ptablect[x][y]; photonToSortAddr++)
            {
                //check elements before photonToSort (curPhotonAddr) until correct spot is found (curPhotonAddr->timestamp < photonToSortAddr->timestamp)
                for(curPhotonAddr = photonToSortAddr-1; curPhotonAddr >= ptable[x][y]; curPhotonAddr--)
                {
                    if(photonToSortAddr->timestamp >= curPhotonAddr->timestamp)
                    {
                        if(curPhotonAddr == photonToSortAddr-1)//this photon is already sorted
                            break;

                        else //moves photonToSort into correct position
                        {
                            photonToSort = *photonToSortAddr;
                            for(photonSwapAddr = photonToSortAddr; photonSwapAddr > curPhotonAddr+1; photonSwapAddr--)
                                *photonSwapAddr = *(photonSwapAddr-1);

                            *(curPhotonAddr+1) = photonToSort;
                            break;

                        }

                    }

                    else if(curPhotonAddr==ptable[x][y]) //Photon is smallest in the table
                    {
                        photonToSort = *photonToSortAddr;
                        for(photonSwapAddr = photonToSortAddr; photonSwapAddr > curPhotonAddr; photonSwapAddr--)
                            *photonSwapAddr = *(photonSwapAddr-1);

                        *curPhotonAddr = photonToSort;
                        break;

                    }

                }

            }


}

void ParseBeamMapFile(char *BeamFile, uint32_t **BeamMap, uint32_t **BeamFlag)
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
        if(flag>0)
            BeamFlag[x][y] = 1;
        else
            BeamFlag[x][y] = 0;
    } while ( ret == 4);    
    fclose(fp);
}

/*
 * Initializes all values of BeamMap to value
 */
void InitializeBeamMap(uint32_t **BeamMap, uint32_t value, int beamCols, int beamRows)
{
    int x, y;
    for(x=0; x<beamCols; x++)
        for(y=0; y<beamRows; y++)
            BeamMap[x][y] = value;
    
}

int main(int argc, char *argv[])
{
    char path[STR_SIZE], fName[STR_SIZE], BeamFile[STR_SIZE], outfile[STR_SIZE], imname[STR_SIZE], tname[STR_SIZE];
    int FirstFile, nFiles,mapflag, beamCols, beamRows, nRoaches;
    long fSize, rd, j, k;
    struct stat st;
    FILE *fp;
    uint64_t **data, *dSize;
    clock_t start, diff, olddiff;
    uint64_t swp,swp1,i,pstart,pcount,tStart,firstHeader, nPhot, tPhot=0;
    struct hdrpacket *hdr;
    char packet[808*16];
    char *olddata;
    uint16_t **image;
    unsigned char *smimage;
    uint64_t *frame;
    uint32_t **BeamMap;
    uint32_t **BeamFlag;
    uint32_t *toWriteBeamMap;
    uint32_t *toWriteBeamFlag;
    uint32_t beamMapInitVal = (uint32_t)(-1);
    char ***ResIdString;
    char addHeaderCmd[120] = "python addH5Header.py ";
    char correctTimestampsCmd[120] = "python correctUnsortedTimestamps.py ";
    photon p1;
    
    photon ***ptable;
    uint32_t **ptablect;

    
    // hdf5 variables
    hid_t file_id;
    hid_t gid_beammap, sid_beammap, did_beammap, did_flagmap, gid_photons, sid_photons, did_photons, dsid_beammap, dsid_flagmap, sid_write_beammap;
    herr_t status;
    hsize_t dims[2];
    hsize_t count[2] = {1, 0};
    hsize_t offset[2] = {0, 0};
    hsize_t stride[2] = {1, 1};
    hsize_t block[2] = {1,1};
    
    size_t dst_size =  sizeof(photon);
    size_t dst_offset[NFIELD] = { HOFFSET( photon, timestamp ), HOFFSET( photon, wvl ), HOFFSET( photon, wSpec ), HOFFSET( photon, wNoise) };
    size_t dst_sizes[NFIELD] = { sizeof(p1.timestamp), sizeof(p1.wvl),sizeof(p1.wSpec), sizeof(p1.wNoise) };
    
    count[1] = beamRows;

    /* Define field information */
    const char *field_names[NFIELD]  =
    { "Time","Wavelength","Spec Weight","Noise Weight"};
    hid_t      field_type[NFIELD];
    hid_t      string_type;
    hsize_t    chunk_size = 10000;
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
    if (ParseConfig(argc,argv,path,&FirstFile,&nFiles,BeamFile,&mapflag,&beamCols,&beamRows) == 0 ) {
        printf("Bin2HDF error - Config parsing error.\n");
		exit(1);
	}

    //initialize nRoaches
    nRoaches = beamRows*beamCols/1000;
    frame = (uint64_t*)malloc(nRoaches*sizeof(uint64_t));
    
    // Set up memory structure for data
    data = (uint64_t **) malloc( nFiles * sizeof(uint64_t *) );
    dSize = (uint64_t *) malloc( nFiles * sizeof(uint64_t *) );

    // Set up memory structure for 2D "beammap" arrays
    BeamMap = (uint32_t**)malloc(beamCols * sizeof(uint32_t*));
    BeamFlag = (uint32_t**)malloc(beamCols * sizeof(uint32_t*));
    image = (uint16_t**)malloc(beamCols * sizeof(uint16_t*));
    smimage = (char *)malloc(beamCols * beamRows * sizeof(char));
    ptable = (photon***)malloc(beamCols * sizeof(photon**));
    ptablect = (uint32_t**)malloc(beamCols * sizeof(uint32_t*));
    ResIdString = (char***)malloc(beamCols * sizeof(char**));
    toWriteBeamMap = (uint32_t*)malloc(beamCols * beamRows * sizeof(uint32_t));
    toWriteBeamFlag = (uint32_t*)malloc(beamCols * beamRows * sizeof(uint32_t));

    for(i=0; i<beamCols; i++)
    {
        BeamMap[i] = (uint32_t*)malloc(beamRows * sizeof(uint32_t));
        BeamFlag[i] = (uint32_t*)malloc(beamRows * sizeof(uint32_t));
        image[i] = (uint16_t*)malloc(beamRows * sizeof(uint16_t));
        ptable[i] = (photon**)malloc(beamRows * sizeof(photon*));
        ptablect[i] = (uint32_t*)malloc(beamRows * sizeof(uint32_t));
        ResIdString[i] = (char**)malloc(beamRows * sizeof(char*));
        for(j=0; j<beamRows; j++)
            ResIdString[i][j] = (char*)malloc(20 * sizeof(char));

    }
        
    // Read in beam map and parse it make 2D beam map and flag arrays
    InitializeBeamMap(BeamMap, beamMapInitVal, beamCols, beamRows); //initialize to out of bounds resID
    InitializeBeamMap(BeamFlag, 1, beamCols, beamRows); //initialize flag to one
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
    dims[0] = beamCols;
    dims[1] = beamRows;
    gid_beammap = H5Gcreate2(file_id, "BeamMap", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    sid_beammap = H5Screate_simple(2, dims, NULL);
    did_beammap = H5Dcreate2(file_id, "/BeamMap/Map", H5T_NATIVE_UINT, sid_beammap, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    did_flagmap = H5Dcreate2(file_id, "/BeamMap/Flag", H5T_NATIVE_UINT, sid_beammap, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    for(i=0; i<beamCols; i++)
        for(j=0; j<beamRows; j++)
        {
            toWriteBeamMap[beamRows*i + j] = BeamMap[i][j];
            toWriteBeamFlag[beamRows*i + j] = BeamFlag[i][j];

        }
    
    H5Dwrite (did_beammap, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, toWriteBeamMap);
    H5Dwrite (did_flagmap, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, toWriteBeamFlag);

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
    for(i=0; i < beamCols; i++) {
		for(j=0; j < beamRows; j++) {
			if( BeamMap[i][j] == 0 ) 
            {
                printf("ResID 0 at (%d,%d)\n", i, j);
                //continue;

            }
            
            if(BeamMap[i][j] == beamMapInitVal) 
            {
                printf("ResID N/A at (%d,%d)\n", i, j);
                continue; 

            }
		
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
        for(j=0; j<beamCols; j++)
            memset(ptablect[j],0,sizeof(uint32_t)*beamRows); // zero out the count table
        
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
                ParsePacket(image, packet, j*8 - pstart, frame, beamCols, beamRows); 
                // add to HDF5 file
                AddPacket(packet,j*8-pstart,file_id,dst_size,dst_offset,dst_sizes,FirstFile,BeamMap,&nPhot,BeamFlag,mapflag,ResIdString,ptable,ptablect,beamCols,beamRows);
		        pstart = j*8;   // move start location for next packet
		        if( pcount%1000 == 0 ) printf("."); fflush(stdout);	                      
            }
        }
        
        //printf("\nSorting photon tables...\n");
        //SortPhotonTables(ptable, ptablect);
        
        // save photon tables to hdf5
        for(j=0; j < beamCols; j++) {
			for(k=0; k < beamRows; k++) {
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
        for( j=0; j < beamCols*beamRows; j++ ) {
            smimage[j] =  (unsigned char) (image[j%beamCols][j/beamCols]/10);
            if( smimage[j] > 2499 ) smimage[j] = 0;
        }
                
        sprintf(imname,"/Images/%ld",i+FirstFile);
        H5IMmake_image_8bit( file_id, imname, (hsize_t)beamCols, (hsize_t)beamRows, smimage );
        for(j=0; j<beamCols; j++)
            memset(image[j], 0, beamRows*sizeof(uint16_t));
        
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
    for(i=0; i < beamCols; i++) {
		for(j=0; j < beamRows; j++) {
			if( BeamMap[i][j] == 0 ) continue;
			free(ptable[i][j]);
		}
	}
    
    for(i=0; i < nFiles; i++) free(data[i]); 
    free(data);
    free(dSize);

    for(i=0; i<beamCols; i++)
    {
        free(BeamMap[i]);
        free(BeamFlag[i]);
        free(image[i]);
        free(ptable[i]);
        free(ptablect[i]);
        free(ResIdString[i]);

    }

    free(BeamMap);
    free(BeamFlag);
    free(image);
    free(ptable);
    free(ptablect);
    free(ResIdString);
    free(toWriteBeamMap);
    free(toWriteBeamFlag);

    strcat(addHeaderCmd, argv[1]);
    strcat(addHeaderCmd, " ");
    strcat(addHeaderCmd, outfile);
    system(addHeaderCmd);
    
    strcat(correctTimestampsCmd, outfile);
    system(correctTimestampsCmd);
    exit(0);

}
