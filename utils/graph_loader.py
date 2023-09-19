import torch
import h5py
from torch_geometric.data import Batch as GraphBatch
from torch_geometric.data import Data as GraphData

def build_graph(filenamelist, device='', network='gravnet'):
    graphs = []
    for filename in filenamelist:
        f = h5py.File(filename, 'r')
        groupname = 'bb0n'
        if 'gamma' in filename:
            groupname = 'gamma'
        group = f[groupname]
        dset_name = group.keys()

        for dset in dset_name:
            df = group[dset]
            dfi = df[:]
            if len(dfi) == 0:
                continue

            dfi[:, 0:2] /= 650. ## x, y scale, unit mm
            dfi[:, 2]   /= 10000.  ## charge scale
            dfi[:, 3]   /= 100.   ## time scale

            tmp_x = torch.tensor(dfi.astype('float32'))
            labelno = 1 if groupname == 'bb0n' else 0
            tmp_y = torch.tensor([labelno])
            tb_graph = GraphData(x=tmp_x, y=tmp_y)
            graphs.append(tb_graph)

        f.close()
    return graphs
             

class GraphData_MoNet(GraphData):
    def __init__(self, h5_path, csv_path, n_channels=2):
        csv_info            = pd.read_csv(csv_path, header=None)
        self.groupname      = np.asarray(csv_info.iloc[:, 0])
        self.attrinfo       = np.asarray(csv_info.iloc[:, 1])
        self.indexinfo      = np.asarray(csv_info.iloc[:, 2])
        self.nodeinfo       = np.asarray(csv_info.iloc[:, 3])
        self.h5file         = h5py.File(h5_path, 'r')
        self.n_channels     = n_channels

    def __len__(self):
        return len(self.groupname)

    def __getitem__(self, idx):
        node = self.h5file[self.groupname[idx]] [self.nodeinfo[idx]]
        attr = self.h5file[self.groupname[idx]] [self.attrinfo[idx]]
        index = self.h5file[self.groupname[idx]] [self.indexinfo[idx]]
        label = 1 if 'bb0n' in self.nodeinfo[idx] else 0
        graph = GraphData(x=node, edge_index=index, edge_attr=attr, y=label)
        return graph















