import numpy as np
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#node_list
metabolite_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\compound_list_final.json'
drug_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\drug_list_final.json'
protein_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\merged_protein_without_PPI.json'
reaction_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\reactions_list_final.json'

#adjacancy_matrix
CR_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\CR_matrix.txt'
DDI_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\DDI_adj_matrix.txt'
DTI_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\DTI_adj_matrix.txt'
PPSS_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\_PPSS_matrix.txt'
PR_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\PR_matrix.txt'
STP_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\STP_adj_matrix.txt'
PPI_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\PPI_matrix.txt'
SPP_matrix_path  = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\SPP_identity_matrix.txt'

#load
with open(metabolite_list_path, "r") as json_file:
    metabolite_list = json.load(json_file)
    mtl = len(metabolite_list)

with open(drug_list_path, "r") as json_file:
    drug_list = json.load(json_file)
    dl = len(drug_list)

with open(protein_list_path, "r") as json_file:
    protein_list = json.load(json_file)
    pl = len(protein_list)

with open(reaction_list_path, "r") as json_file:
    reaction_list = json.load(json_file)
    rl = len(reaction_list)

print(f"metabolite:{mtl}, drug:{dl}, protein:{pl}, reaction:{rl}")

CR_matrix = np.loadtxt(CR_matrix_path)
DDI_matrix = np.loadtxt(DDI_matrix_path)
TDI_matrix = np.loadtxt(DTI_matrix_path)
PP_matrix = np.loadtxt(PPI_matrix_path)
PR_matrix = np.loadtxt(PR_matrix_path)
STP_matrix = np.loadtxt(STP_matrix_path)
RC_matrix = CR_matrix.T
DTI_matrix = TDI_matrix.T
RP_matrix = PR_matrix.T

#use STP and PP temporarily before similarity matrix is available
print(f"shapes: CR_matrix: {CR_matrix.shape}, RC_matrix: {RC_matrix.shape}, DDI_matrix: {DDI_matrix.shape}, DTI_matrix: {DTI_matrix.shape}, TDI_matrix: {TDI_matrix.shape}, PPI_matrix: {PP_matrix.shape}, PR_matrix: {PR_matrix.shape}, RP_matrix: {RP_matrix.shape}, STP_matrix: {STP_matrix.shape}")

heo_orders = ['PP', 'SPP', 'PPxPP', 'SPPxPP', 'PPxSPP', 'SPPxSPP', 'PRxRP', 'PPxPRxRP', 'SPPxPRxRP', 'PRxRPxPP', 'PRxRPxSPP', 'PRxRPxPRxRP', 'PRxRCxCRxRP', 'PPxPRxRCxCRxRP', 'SPPxPRxRCxCRxRP', 'PRxRCxCRxRPxPP', 'PRxRCxCRxRPxSPP']
bio_orders = ['DT', 'RP', 'TD', 'PR', 'DTxPP', 'DTxSPP', 'DDxDT', 'CRxRP', 'RPxPP', 'RPxSPP', 'PPxTD', 'SPPxTD', 'PPxPR', 'SPPxPR', 'TDxDT', 'TDxDD', 'PRxRC', 'DTxPPxPP', 'DTxSPPxPP', 'DTxPPxSPP', 'DTxSPPxSPP', 'DTxTDxDT', 'DTxPRxRP', 'DDxDTxPP', 'DDxDTxSPP', 'CRxRPxPP', 'CRxRPxSPP', 'RPxPPxPP', 'RPxSPPxPP', 'RPxPPxSPP', 'RPxSPPxSPP', 'RPxTDxDT', 'RPxPRxRP', 'RCxCRxRP', 'PPxPPxTD', 'SPPxPPxTD', 'PPxSPPxTD', 'SPPxSPPxTD', 'PPxPPxPR', 'SPPxPPxPR', 'PPxSPPxPR', 'SPPxSPPxPR', 'PPxTDxDT', 'SPPxTDxDT', 'PPxTDxDD', 'SPPxTDxDD', 'PPxPRxRC', 'SPPxPRxRC', 'TDxDTxPP', 'TDxDTxSPP', 'TDxDTxTD', 'TDxDTxPR', 'TDxDDxDT', 'PRxRPxTD', 'PRxRPxPR', 'PRxRCxCR', 'DTxPPxTDxDT', 'DTxSPPxTDxDT', 'DTxPPxPRxRP', 'DTxSPPxPRxRP', 'DTxTDxDTxPP', 'DTxTDxDTxSPP', 'DTxPRxRPxPP', 'DTxPRxRPxSPP', 'DDxDTxPPxPP', 'DDxDTxSPPxPP', 'DDxDTxPPxSPP', 'DDxDTxSPPxSPP', 'DDxDTxPRxRP', 'CRxRPxPPxPP', 'CRxRPxSPPxPP', 'CRxRPxPPxSPP', 'CRxRPxSPPxSPP', 'CRxRPxPRxRP', 'RPxPPxTDxDT', 'RPxSPPxTDxDT', 'RPxPPxPRxRP', 'RPxSPPxPRxRP', 'RPxTDxDTxPP', 'RPxTDxDTxSPP', 'RPxTDxDDxDT', 'RPxPRxRPxPP', 'RPxPRxRPxSPP', 'RCxCRxRPxPP', 'RCxCRxRPxSPP', 'PPxPPxTDxDD', 'SPPxPPxTDxDD', 'PPxSPPxTDxDD', 
'SPPxSPPxTDxDD', 'PPxPPxPRxRC', 'SPPxPPxPRxRC', 'PPxSPPxPRxRC', 'SPPxSPPxPRxRC', 'PPxTDxDTxTD', 'SPPxTDxDTxTD', 'PPxTDxDTxPR', 'SPPxTDxDTxPR', 'PPxTDxDDxDT', 'SPPxTDxDDxDT', 'PPxPRxRPxTD', 'SPPxPRxRPxTD', 'PPxPRxRPxPR', 'SPPxPRxRPxPR', 'PPxPRxRCxCR', 'SPPxPRxRCxCR', 'TDxDTxPPxTD', 'TDxDTxSPPxTD', 'TDxDTxPPxPR', 'TDxDTxSPPxPR', 'TDxDTxTDxDT', 'TDxDTxPRxRP', 'TDxDDxDTxPP', 'TDxDDxDTxSPP', 'TDxDDxDTxPR', 'PRxRPxPPxTD', 'PRxRPxSPPxTD', 'PRxRPxPPxPR', 'PRxRPxSPPxPR', 'PRxRPxTDxDT', 'PRxRPxTDxDD', 'PRxRPxPRxRC', 'DTxTDxDTxPRxRP', 'DTxPRxRPxTDxDT', 'DTxPRxRPxPRxRP', 'DDxDTxPPxPRxRP', 'DDxDTxSPPxPRxRP', 'DDxDTxPRxRPxPP', 'DDxDTxPRxRPxSPP', 'CRxRPxPPxPRxRP', 'CRxRPxSPPxPRxRP', 'CRxRPxPRxRPxPP', 'CRxRPxPRxRPxSPP', 'RPxPPxTDxDDxDT', 'RPxSPPxTDxDDxDT', 'RPxTDxDTxTDxDT', 'RPxTDxDTxPRxRP', 'RPxTDxDDxDTxPP', 'RPxTDxDDxDTxSPP', 'RPxPRxRPxTDxDT', 'RCxCRxRPxPPxPP', 'RCxCRxRPxSPPxPP', 'RCxCRxRPxPPxSPP', 'RCxCRxRPxSPPxSPP', 'PPxPPxPRxRCxCR', 'SPPxPPxPRxRCxCR', 'PPxSPPxPRxRCxCR', 'SPPxSPPxPRxRCxCR', 'PPxTDxDDxDTxPR', 'SPPxTDxDDxDTxPR', 'PPxPRxRPxTDxDD', 'SPPxPRxRPxTDxDD', 'PPxPRxRPxPRxRC', 'SPPxPRxRPxPRxRC', 'TDxDTxTDxDTxPR', 'TDxDTxPRxRPxTD', 'TDxDTxPRxRPxPR', 'TDxDDxDTxPPxPR', 'TDxDDxDTxSPPxPR', 'TDxDDxDTxPRxRP', 'PRxRPxPPxTDxDD', 'PRxRPxSPPxTDxDD', 'PRxRPxPPxPRxRC', 'PRxRPxSPPxPRxRC', 'PRxRPxTDxDTxTD', 'PRxRPxTDxDTxPR', 'PRxRPxTDxDDxDT', 'PRxRPxPRxRPxTD', 'DDxDTxPRxRPxPRxRP', 'RPxTDxDDxDTxPRxRP', 'RPxPRxRPxTDxDDxDT', 'TDxDDxDTxPRxRPxPR', 'PRxRPxTDxDDxDTxPR', 'PRxRPxPRxRPxTDxDD']


matrices = {
    'PP': PP_matrix,
    'SPP': STP_matrix,
    'PR': PR_matrix,
    'RP': RP_matrix,
    'CR': CR_matrix,
    'DD': DDI_matrix,
    'DT': DTI_matrix,
    'TD': TDI_matrix,
    'RC': RC_matrix
}

#heo
for i in heo_orders:
    x = i.split('x')
    if len(x) == 1:
        final_matrix = matrices[x[0]]
    else:
        for j in range(len(x)):
            if j == 0:
                final_matrix = matrices[x[j]]
            else:
                final_matrix = np.dot(final_matrix, matrices[x[j]])


    np.savetxt(r'D:\ISEF\HetMGNN\heometapaths\\'+i, final_matrix, fmt='%.3f')
    base_cmap = plt.cm.spring
    colors = [(0, 0, 0)] + [base_cmap(x) for x in np.linspace(0.001, np.max(final_matrix), 256)]
    cmap_name = 'custom_spring'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Plot and save the heatmap
    plt.imshow(final_matrix, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig(r'D:\ISEF\HetMGNN\heometapaths\\' + i + '.png', dpi=1000)
    plt.close()


for i in bio_orders:
    x = i.split('x')
    if len(x) == 1:
        final_matrix = matrices[x[0]]
    else:
        for j in range(len(x)):
            if j == 0:
                final_matrix = matrices[x[j]]
            else:
                final_matrix = np.dot(final_matrix, matrices[x[j]])


    np.savetxt(r'D:\ISEF\HetMGNN\biometapaths\\'+i, final_matrix, fmt='%.3f')
    base_cmap = plt.cm.spring
    colors = [(0, 0, 0)] + [base_cmap(x) for x in np.linspace(0.001, np.max(final_matrix), 256)]
    cmap_name = 'custom_spring'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Plot and save the heatmap
    plt.imshow(final_matrix, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig(r'D:\ISEF\HetMGNN\biometapaths\\' + i + '.png', dpi=1000)
    plt.close()

            

























