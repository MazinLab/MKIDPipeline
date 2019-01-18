typedef struct photon {
    uint32_t resID;
    uint32_t timestamp;
    float wvl;
    float wSpec;
    float wNoise;
} photon;

long extract_photons(const char *dname, unsigned long start, unsigned long inttime,
                     const char *bmap, unsigned int bmap_ncol, unsigned int bmap_nrow,
                     unsigned long n_max_photons, photon* photons);

long extract_photons_dummy(const char *dname, unsigned long start, unsigned long inttime,
                     const char *bmap, unsigned int bmap_ncol, unsigned int bmap_nrow,
                     unsigned long n_max_photons, photon* photons);