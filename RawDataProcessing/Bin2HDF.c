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
#define NFIELD 6

// MKID array stats
#define NROACHES 10
#define NPIXELS_PER_ROACH 1024
#define BEAM_ROWS 125
#define BEAM_COLS 80
#define RAD2DEG 57.2957795131

// kludge for Palomar run 4/17 when roach id not set right
uint32_t roachidarr[10000] = {0}, residarr[10000] = {0};
uint64_t tstart[10] = {0};
uint64_t rframe[10] = {0};

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
    uint32_t resid;
    uint16_t x;
    uint16_t y;
    uint32_t timestamp;
    float baseline;
    float wvl;
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

int find_index(int value)
{
   int i;
   for (i=0; i<10000; i++)
   {
	 if (residarr[i] == value)
	 {
	    return(i);  /* it was found */
	 }
   }
   return(-1);  /* if it was not found */
}

void FindStart(char *data, uint32_t BeamMap[BEAM_COLS][BEAM_ROWS])
{
    uint64_t i,swp,swp1,swp2,swp3;
    int64_t basetime;
    uint32_t rid,roach;
    struct hdrpacket *hdr;
    struct datapacket *data1;
    
     
    // look at first data packet to determine what roach this packet came from by resid - kludge for 4/17 Palomar run
    for (i=0; i < 1e6/8; i++) {
        // get info from header packet
        swp = *((uint64_t *) (&data[i*8]));
        swp1 = __bswap_64(swp);
        hdr = (struct hdrpacket *) (&swp1);       
        
        // check for a new start packet
        if (hdr->start == 0b11111111) {
            swp2 = *((uint64_t *) (&data[(i+1)*8]));
            swp3 = __bswap_64(swp2);
            data1 = (struct datapacket *) (&swp3);   

            rid = BeamMap[data1->xcoord%BEAM_COLS][data1->ycoord%BEAM_ROWS];
            roach = roachidarr[ find_index(rid) ];
            if( roach == -1 ) { printf("Error finding ROACH ID\n"); roach=0;}
    
            // if no start timestamp, store start timestamp
            if( tstart[roach] > hdr->timestamp ) tstart[roach] = (uint64_t) hdr->timestamp;
        }
    }
}

void AddPacket(char *packet, uint64_t l, hid_t file_id, size_t dst_size, size_t dst_offset[NFIELD], size_t dst_sizes[NFIELD], int FirstFile, uint32_t BeamMap[BEAM_COLS][BEAM_ROWS], photon *p, uint64_t *nPhot, uint32_t BeamFlag[BEAM_COLS][BEAM_ROWS], int mapflag)
{
    uint64_t i,swp,swp1,swp2,swp3;
    int64_t basetime;
    uint32_t rid,roach;
    struct hdrpacket *hdr;
    struct datapacket *data;
    
    // get info form header packet
    swp = *((uint64_t *) (&packet[0]));
    swp1 = __bswap_64(swp);
    hdr = (struct hdrpacket *) (&swp1);             
    if (hdr->start != 0b11111111) {
        printf("Error - packet does not start with a correctly formatted header packet!\n");
        //printf(".");
        return;
    }
    
    // look at first data packet to determine what roach this packet came from by resid - kludge for 4/17 Palomar run
    swp2 = *((uint64_t *) (&packet[8]));
    swp3 = __bswap_64(swp2);
    data = (struct datapacket *) (&swp3);   
    rid = BeamMap[data->xcoord%BEAM_COLS][data->ycoord%BEAM_ROWS];
    roach = roachidarr[ find_index(rid) ];
    if( roach == -1 ) { printf("Error finding ROACH ID\n"); roach=0;}
    
    // if no start timestamp, store start timestamp
    if( tstart[roach] == 0 ) tstart[roach] = (uint64_t) hdr->timestamp;
    basetime = hdr->timestamp - tstart[roach]; // time since start of first file
    if( basetime < 0 ) basetime = 0; // maybe have some packets out of order early in file
    
    // check for lost packets
    //if( rframe[roach] == 0 ) rframe[roach] = hdr->frame;
    //if( hdr->frame != (rframe[roach]+1)%4096 ) printf("id %d roach %d prev %d cur %d\n",rid,roach,(rframe[roach]+1)%4096,hdr->frame);
    //rframe[roach] = hdr->frame;
    
    // Load packets into memory then append all the photons in the packet to the h5 file at once
    //printf("Header: %16lx : %d %d %ld\n", (uint64_t) swp1, hdr->start, hdr->roach, (uint64_t) hdr->timestamp);
    //printf("Header: %16lx : %d %d %d %ld\n", (uint64_t) swp1, hdr->start, roach, hdr->frame, (uint64_t) hdr->timestamp); fflush(stdout);
    //if( *nPhot < 5e3) printf("ts: %d %d %ld %ld %ld\n",rid,roach,tstart[roach],hdr->timestamp,basetime);
    //printf("ROACH: %d Timestamp: %ld\n", hdr->roach, (uint64_t) hdr->timestamp);
    
    for(i=1;i<l/8;i++) {
       
       swp = *((uint64_t *) (&packet[i*8]));
       swp1 = __bswap_64(swp);
       data = (struct datapacket *) (&swp1);
       if( data->xcoord >= BEAM_COLS || data->ycoord >= BEAM_ROWS ) continue;
       if( mapflag > 0 && BeamFlag[data->xcoord][data->ycoord] > 0) continue ; // if mapflag is set only record photons that were succesfully beammapped       

       //printf("Data: %d %d %d %d %d\n", data->xcoord, data->ycoord, data->timestamp, data->baseline, data->wvl);
       //exit(0);      
       //if( *nPhot < 200) printf("photon: %d %d\n",data->baseline,data->wvl);
       
       p[*nPhot].resid = BeamMap[data->xcoord][data->ycoord];
       p[*nPhot].x = data->xcoord;
       p[*nPhot].y = data->ycoord;
       p[*nPhot].timestamp = (uint32_t) (basetime*500 + data->timestamp);
       p[*nPhot].baseline = ((float) data->baseline)*RAD2DEG/16384.0;
       p[*nPhot].wvl = ((float) data->wvl)*RAD2DEG/32768.0;
       (*nPhot)++;
   
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
    char path[STR_SIZE], fName[STR_SIZE], BeamFile[STR_SIZE], outfile[STR_SIZE], imname[STR_SIZE];
    int FirstFile, nFiles,mapflag;
    long fSize, rd, j;
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
    photon *p, p1;
    
    // hdf5 variables
    hid_t file_id;
    hid_t gid_beammap, sid_beammap, did_beammap, did_flagmap, gid_photons, sid_photons, did_photons;
    herr_t status;
    hsize_t dims[2];

    size_t dst_size =  sizeof(photon);
    size_t dst_offset[NFIELD] = {   HOFFSET( photon, resid ), HOFFSET( photon, x), HOFFSET( photon, y ), 
                                    HOFFSET( photon, timestamp ), HOFFSET( photon, baseline ), HOFFSET( photon, wvl ) };
    size_t dst_sizes[NFIELD] = { sizeof(p1.resid), sizeof(p1.x),sizeof(p1.y), sizeof(p1.timestamp), 
                                 sizeof(p1.baseline), sizeof(p1.wvl) };

    /* Define field information */
    const char *field_names[NFIELD]  =
    { "ResID","X","Y","Time","Baseline", "Wavelength"};
    hid_t      field_type[NFIELD];
    hid_t      string_type;
    hsize_t    chunk_size = 10;
    int        *fill_data = NULL;
    int        compress  = 0;

    /* Initialize field_type */
    field_type[0] = H5T_STD_U32LE;
    field_type[1] = H5T_STD_U16LE;
    field_type[2] = H5T_STD_U16LE;                                                  
    field_type[3] = H5T_STD_U32LE;
    field_type[4] = H5T_NATIVE_FLOAT;
    field_type[5] = H5T_NATIVE_FLOAT;      
        
    memset(packet, 0, sizeof(packet[0]) * 808 * 16);    // zero out array
    
    // Open config file and parse    
    if( argc != 2 ) {
        printf("BinToHDF error - First command line argument must be the configuration file.\n");
        exit(0);
    }
    if (ParseConfig(argc,argv,path,&FirstFile,&nFiles,BeamFile,&mapflag) == 0 ) exit(1);
    
    // Set up memory structure for data
    data = (uint64_t **) malloc( nFiles * sizeof(uint64_t *) );
    dSize = (uint64_t *) malloc( nFiles * sizeof(uint64_t *) );

    // Read in files to kludge in ROACH ID since firmware wasn't outputting it during the 4/17 Palomar run
    fp = fopen("/mnt/data0/bmazin/roachid.txt","r");
    for(i=0; i < 10000; i++) {
        fscanf(fp," %d %d\n", &residarr[i], &roachidarr[i]);
    }
    fclose(fp);
    //printf("resid, roachid: %d %d\n",residarr[0],roachidarr[0]);
    //printf("resid, roachid: %d %d\n",residarr[1],roachidarr[1]);
        
    // Read in beam map and parse it make 2D beam map and flag arrays
    ParseBeamMapFile(BeamFile,BeamMap,BeamFlag);
    
    // Read all the data into memory
    start = clock();
    for(i=0; i < nFiles; i++) {
        sprintf(fName,"%s/%d.bin",path,FirstFile+i);
        stat(fName, &st);
        fSize = st.st_size;
        printf("Reading %s - %d bytes\n",fName,fSize);
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
    
    // Step through .bin files that are now in memory, suck out the data, convert it into ARCON packet format, 
    // and write it to the h5 file
    start = clock();
  
    gid_photons = H5Gcreate2(file_id, "Photons", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5TBmake_table( "Photon Data", file_id, "/Photons/data", NFIELD, 0, dst_size, field_names, dst_offset, field_type,chunk_size, fill_data, compress, &p);
    
    // Kludge to find start times since not recorded right in 4/17 Palomar .bin files
    //FindStart((char *)data[0], BeamMap);
    
    for(i=0; i < nFiles; i++) {
        olddata = (char *) data[i];
        pstart = 0;
        pcount = 0;
        nPhot = 0;
        
        printf("File %d: ",i);
        p = (photon *) malloc( sizeof(photon) * dSize[i]/8 );  // allocate array to hold photons
        
        // .bin may not always start with a header packet, so search until we find the first header
        for( j=0; j<dSize[i]/8; j++) {
            swp = *((uint64_t *) (&olddata[j*8]));
            swp1 = __bswap_64(swp);
            hdr = (struct hdrpacket *) (&swp1);
            if (hdr->start == 0b11111111) {
                firstHeader = j;
                pstart = j;
                if( firstHeader != 0 ) printf("First header at %d\n",firstHeader);
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
                AddPacket(packet,j*8-pstart,file_id,dst_size,dst_offset,dst_sizes,FirstFile,BeamMap,p,&nPhot,BeamFlag,mapflag);
		        pstart = j*8;   // move start location for next packet
		        if( pcount%1000 == 0 ) printf("."); fflush(stdout);	                      
            }
        }
        
        // save photon data to hdf5
        H5TBappend_records(file_id,  "/Photons/data", nPhot, dst_size, dst_offset, dst_sizes, p );  
        free(p);
      
        
        // save image array to hdf5
        for( j=0; j < BEAM_COLS*BEAM_ROWS; j++ ) {
            smimage[j] =  (unsigned char) (image[j%BEAM_COLS][j/BEAM_COLS]/10);
            if( smimage[j] > 2499 ) smimage[j] = 0;
        }
                
        sprintf(imname,"/Images/%d",i+FirstFile);
        H5IMmake_image_8bit( file_id, imname, (hsize_t)BEAM_COLS, (hsize_t)BEAM_ROWS, smimage );
        memset(image, 0, BEAM_COLS*BEAM_ROWS*sizeof(uint16_t));
        
        printf(" %d packets, %d photons. %d photons/packet.\n",pcount,nPhot,nPhot/pcount);
        tPhot += nPhot;
    }

    H5Gclose(gid_photons);   
    
    diff = clock()-start;
    printf("Parsed %ld photons in %f seconds: %9.1f photons/sec.\n",tPhot,(float)(diff+olddiff)/CLOCKS_PER_SEC,((float)tPhot)/((float)(diff+olddiff)/CLOCKS_PER_SEC));

    // Close up
    H5Gclose(gid_beammap);   
    H5Fclose(file_id);
    
    for(i=0; i < nFiles; i++) free(data[i]); 
    free(data);
    free(dSize);

}
