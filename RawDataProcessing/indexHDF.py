import tables
import sys, os

if len(sys.argv)<2:
    print('Must specify H5 filename')
    exit(0)

hfile = tables.open_file(sys.argv[1], 'a')
hfile.set_node_attr('/', 'PYTABLES_FORMAT_VERSION', '2.0')
hfile.format_version = '2.0'

#filterObj = tables.Filters(complevel=5, complib='lzo')
filterObj = tables.Filters(complevel=0, complib='lzo')

print('Opened file')

# for photonTable in hfile.iter_nodes('/Photons'):
#     print('Indexing table', photonTable)
#     photonTable.cols.Time.create_index()
#     photonTable.cols.Wavelength.create_index()
#     photonTable.flush()

photonTable = hfile.root.Photons.PhotonTable
photonTable.cols.Time.create_csindex(filters=filterObj)
photonTable.cols.ResID.create_csindex(filters=filterObj)
photonTable.cols.Wavelength.create_csindex(filters=filterObj)
photonTable.flush()

hfile.close()
