
long parsebin(const char* file, unsigned long a_len, double* a1, double* a2) {

    unsigned long i=0,out_i=0;
    bool more_recs=true;

    //open up file

    //if not open return -1

    while(more_recs) {
        //parse next data

        //store data, will overwrite last entry once arrays are full
        a1[out_i]=1337;
        a2[out_i]=1337;
        more_recs = i<10 ? true:false;
        i++;
        if (i<a_len) out_i=i;
    }

    //close up file

    return i;
}
