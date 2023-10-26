# load general packages and functions
# (none)
import numpy as np
# load program-specific functions
from parameters.constants import constants as C
import gnn.mpnn
import util
import torch
from torch import nn 
# from Workflow import Workflow
import generate
import rdkit
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import random
# defines the models with parameters from `constants.py`


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.generative_model = gnn.mpnn.GGNN(
            f_add_elems=C.dim_f_add_p1,
            edge_features=C.dim_edges[2],
            enn_depth=C.enn_depth,
            enn_dropout_p=C.enn_dropout_p,
            enn_hidden_dim=C.enn_hidden_dim,
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_depth=C.mlp2_depth,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            gather_att_depth=C.gather_att_depth,
            gather_att_dropout_p=C.gather_att_dropout_p,
            gather_att_hidden_dim=C.gather_att_hidden_dim,
            gather_width=C.gather_width,
            gather_emb_depth=C.gather_emb_depth,
            gather_emb_dropout_p=C.gather_emb_dropout_p,
            gather_emb_hidden_dim=C.gather_emb_hidden_dim,
            hidden_node_features=C.hidden_node_features,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            message_size=C.message_size,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
        )
        self.connect_model = gnn.mpnn.Connect_nodes(
            f_add_elems=C.dim_f_add_p1,
            edge_features=C.dim_edges[2],
            enn_depth=C.enn_depth,
            enn_dropout_p=C.enn_dropout_p,
            enn_hidden_dim=C.enn_hidden_dim,
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_depth=C.mlp2_depth,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            gather_att_depth=C.gather_att_depth,
            gather_att_dropout_p=C.gather_att_dropout_p,
            gather_att_hidden_dim=C.gather_att_hidden_dim,
            gather_width=C.gather_width,
            gather_emb_depth=C.gather_emb_depth,
            gather_emb_dropout_p=C.gather_emb_dropout_p,
            gather_emb_hidden_dim=C.gather_emb_hidden_dim,
            hidden_node_features=C.hidden_node_features,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            message_size=C.message_size,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
        )

    def forward(self, linker_nodes,linker_edges, fragment_nodes, fragment_edges,is_train=False,**kw):
        apd_output,_,_ = self.generative_model(linker_nodes,linker_edges, fragment_nodes, fragment_edges)       
        apd_output = apd_output
        tanimoto_loss_list = []
        if is_train:
            fragments_list = kw['fragment_smi_list']
            ground_truth_list = kw['grountruth_smi_list']
            epoch = kw['epoch']
        # fragments_list,ground_truth_list = smi_list
            g, a, f, t,generated_nodes,generated_edges,generated_n_nodes,_ = generate.build_graphs(model=self.generative_model,
                                                n_graphs_to_generate=C.batch_size,
                                                batch_size=C.batch_size)
            
            generated_n_nodesss = generated_n_nodes
            generated_n_nodesss = torch.tensor(generated_n_nodesss).to(torch.device("cuda"))
            n_graphs_to_generate = C.n_samples
        
        
            smi_list = []

            for idx, molecular_graph in enumerate(g):

                mol = molecular_graph.get_molecule()
                smi = Chem.MolToSmiles(mol)
                smi_list.append(smi)
                try:
                    mol.UpdatePropertyCache(strict=False)
                    rdkit.Chem.SanitizeMol(mol)
                    
                    
                except (ValueError, RuntimeError, AttributeError):
                    pass
            generated_nodes = torch.tensor(generated_nodes).to(torch.device("cuda"))
            generated_edges = torch.tensor(generated_edges).to(torch.device("cuda"))
       

       
        connect_out,_,_ = self.connect_model(linker_nodes,linker_edges)
        _,two_idx = torch.topk(connect_out, k=2, dim=1, largest=True)
       
        if is_train:
            for id,atom_idx in enumerate(two_idx):
                try:
                    fragments_smi = fragments_list[id]
                    fragments = Chem.MolFromSmiles(fragments_smi)
                    linker_smi_1 = smi_list[id]
                    
                    linker = Chem.MolFromSmiles(linker_smi_1)
                    #connect
                    combo = Chem.CombineMols(fragments,linker)
                    idx_list = []
                    for atom in fragments.GetAtoms():
                        if atom.GetSymbol() == '*':
                            f2 = atom.GetIdx()
                            idx_list.append(f2)

                    du = Chem.MolFromSmiles('*')
                    combo = AllChem.DeleteSubstructs(combo,du)
                    edcombo = Chem.EditableMol(combo)
                    try:
                        edcombo.AddBond(idx_list[0],atom_idx[0]+idx_list[1]-2,order=Chem.rdchem.BondType.SINGLE)
                        edcombo.AddBond(idx_list[1]-2,atom_idx[1]+idx_list[1]-2,order=Chem.rdchem.BondType.SINGLE)
                        final_connect_mol = edcombo.GetMol()
                    except:
                
                        fragment_smi_0 = fragments_smi.split('.')[0]
                        fragment_smi_1 = fragments_smi.split('.')[1]
                        
                        match = '****sfs'
                        for _ in range(1000):
                            if not Chem.MolFromSmiles(match):
                                s = linker_smi_1
                                inStr = "(*)"
                                for i in range(2):
                                    index = random.randint(0, len(s))
                                    s = "".join([s[:index], inStr, s[index:]])
                                match = s
                            else:
                                break
                        smi_linker = []
                        for _ in range(1000):
                            smi_link = Chem.MolToSmiles(Chem.MolFromSmiles(s), doRandom=True)
                            if smi_link[0] == "*" and smi_link[-1] == '*':
                                smi_linker.append(smi_link)
                        
                            
                        final_smi = (fragment_smi_1 + smi_linker[0] + fragment_smi_0).replace("*", "")

                        final_connect_mol = Chem.MolFromSmiles(final_smi)
                        
                    final_connect_smi = Chem.MolToSmiles(final_connect_mol)
                       
                    #calculate their tanimoto similarities
                    k=0.8
                    ground_truth_smi = ground_truth_list[id]
                    query_mol = Chem.MolFromSmiles(ground_truth_smi)
                    query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)
                    fp = AllChem.GetMorganFingerprint(final_connect_mol, 2, useCounts=True, useFeatures=True)
                    tanimoto_score = DataStructs.TanimotoSimilarity(query_fp, fp)
                    tanimoto_loss =1 - (min(tanimoto_score, k) / k)
                    tanimoto_loss_list.append(tanimoto_loss)
                except:
                    print('error')
                    tanimoto_loss_list.append(1)
            tanimoto_tensor = torch.tensor(tanimoto_loss_list).to(torch.device("cuda"))
        else:
            tanimoto_tensor=torch.tensor(1)
        return apd_output,tanimoto_tensor,two_idx




    

def graph_to_mol(node_features, edge_features, n_nodes):
    """ Converts input graph represenetation (node and edge features) into an
    `rdkit.Mol` object.

    Args:
      node_features (torch.Tensor) : Node features tensor.
      edge_features (torch.Tensor) : Edge features tensor.
      n_nodes (int) : Number of nodes in the graph representation.

    Returns:
      molecule (rdkit.Chem.Mol) : Molecule object.
    """
    # create empty editable `rdkit.Chem.Mol` object
    molecule = rdkit.Chem.RWMol()
    node_to_idx = {}

    # add atoms to editable mol object
    for v in range(n_nodes):
        atom_to_add = features_to_atom(v, node_features)
        molecule_idx = molecule.AddAtom(atom_to_add)
        node_to_idx[v] = molecule_idx

    # add bonds to atoms in editable mol object; to not add the same bond twice
    # (which leads to an error), mask half of the edge features beyond diagonal
    n_max_nodes = C.dim_nodes[0]
    edge_mask = torch.triu(
        torch.ones((n_max_nodes, n_max_nodes), device="cuda"), diagonal=1
    )
    edge_mask = edge_mask.view(n_max_nodes, n_max_nodes, 1)
    edges_idc = torch.nonzero(edge_features * edge_mask)

    for vi, vj, b in edges_idc:
        molecule.AddBond(
            node_to_idx[vi.item()],
            node_to_idx[vj.item()],
            C.int_to_bondtype[b.item()],
        )

    # convert editable mol object to non-editable mol object
    try:
        molecule.GetMol()
    except AttributeError:  # will throw an error if molecule is `None`
        pass

    # correct for ignored Hs
    if C.ignore_H and molecule:
        try:
            rdkit.Chem.SanitizeMol(molecule)
        except ValueError:
            # throws 1st exception if "molecule" is too ugly to be corrected
            pass

    return molecule

def features_to_atom(node_idx, node_features):
    """ Converts the node feature vector corresponding to the specified node
    into an atom object.

    Args:
      node_idx (int) : Index denoting the specific node on the graph to convert.
      node_features (torch.Tensor) : Node features tensor for one graph.

    Returns:
      new_atom (rdkit.Atom) : Atom object corresponding to specified node
        features.
    """
    # get all the nonzero indices in the specified node feature vector
    nonzero_idc = torch.nonzero(node_features[node_idx])

    # determine atom symbol
    atom_idx = nonzero_idc[0]
    atom_type = C.atom_types[atom_idx]

    # initialize atom
    new_atom = rdkit.Chem.Atom(atom_type)

    # determine formal charge
    fc_idx = nonzero_idc[1] - C.n_atom_types
    formal_charge = C.formal_charge[fc_idx]

    new_atom.SetFormalCharge(formal_charge)  # set property

    # determine number of implicit Hs (if used)
    if not C.use_explicit_H and not C.ignore_H:
        total_num_h_idx = nonzero_idc[2] - C.n_atom_types - C.n_formal_charge
        total_num_h = C.imp_H[total_num_h_idx]

        new_atom.SetUnsignedProp("_TotalNumHs", total_num_h)  # set property
    elif C.ignore_H:
        # these will be set with structure is "sanitized" (corrected) later
        # in `mol_to_graph()`.
        pass

    # determine chirality (if used)
    if C.use_chirality:
        cip_code_idx = (
            nonzero_idc[-1]
            - C.n_atom_types
            - C.n_formal_charge
            - bool(not C.use_explicit_H and not C.ignore_H) * C.n_imp_H
        )
        cip_code = C.chirality[cip_code_idx]
        new_atom.SetProp("_CIPCode", cip_code)  # set property

    return new_atom




def initialize_model():
    """ Initializes the model to be trained. Possible model: "GGNN".

    Returns:
      model (modules.SummationMPNN or modules.AggregationMPNN or
        modules.EdgeMPNN) : Neural net model.
    """
    try: 
        hidden_node_features = C.hidden_node_features
    except AttributeError:  # raised for EMN model only
        hidden_node_features = None
        edge_emb_hidden_dim = C.edge_emb_hidden_dim

    if C.model == "GGNN":
        net1 = Model()
     

    else:
        raise NotImplementedError("Model is not defined.")

    # net1 = net1.to("cuda", non_blocking=True)
    net1 = net1.cuda()
    

    return net1
