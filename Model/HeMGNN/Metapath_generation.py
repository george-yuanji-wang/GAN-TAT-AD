from collections import Counter

compatibility = {
    'D':['P', 'D'],
    'C':['R'],
    'R':['P', 'C'],
    'P':['P', 'D', 'R']
}

metapaths1 = ['D', 'C', 'R', 'P']

def longer_metapath(metapaths):
    new_metapaths = []
    for metapath in metapaths:
        last = metapath[-1]
        options = compatibility[last]
        for t in options:
            new_metapaths.append(metapath + '-' + t)

    return new_metapaths


metapaths2 = longer_metapath(metapaths1)
metapaths3 = longer_metapath(metapaths2)
metapaths4 = longer_metapath(metapaths3)
metapaths5 = longer_metapath(metapaths4)
metapaths6 = longer_metapath(metapaths5)
metapaths7 = longer_metapath(metapaths6)
metapaths8 = longer_metapath(metapaths7)
metapaths9 = longer_metapath(metapaths8)
metapaths10 = longer_metapath(metapaths9)
metapaths11 = longer_metapath(metapaths10)
metapaths = metapaths1 + metapaths2 + metapaths3 + metapaths4 + metapaths5 + metapaths6 + metapaths7 + metapaths8 + metapaths9 + metapaths10 + metapaths11

filtered_metapaths = []

for metapath in metapaths:
    nodes = metapath.split('-')
    count = Counter(nodes)
    if nodes[0] == 'P' or nodes[-1] == 'P':
        if len(set(nodes)) <= 3:
            if count['P'] <= 3:
                if count['D'] <= 2 and count['R']<=2 and count['C'] <=1:
                    filtered_metapaths.append(metapath)

metapathss = ['P', 'D-P', 'R-P', 'P-P', 'P-D', 'P-R', 'D-P-P', 'D-D-P', 'C-R-P', 'R-P-P', 'P-P-P', 'P-P-D', 'P-P-R', 'P-D-P', 'P-D-D', 'P-R-P', 'P-R-C', 'D-P-P-P', 'D-P-D-P', 
'D-P-R-P', 'D-D-P-P', 'C-R-P-P', 'R-P-P-P', 'R-P-D-P', 'R-P-R-P', 'R-C-R-P', 'P-P-P-D', 'P-P-P-R', 'P-P-D-P', 'P-P-D-D', 'P-P-R-P', 'P-P-R-C', 'P-D-P-P', 'P-D-P-D', 'P-D-P-R', 
'P-D-D-P', 'P-R-P-P', 'P-R-P-D', 'P-R-P-R', 'P-R-C-R', 'D-P-P-D-P', 'D-P-P-R-P', 'D-P-D-P-P', 'D-P-R-P-P', 'D-D-P-P-P', 'D-D-P-R-P', 'C-R-P-P-P', 'C-R-P-R-P', 'R-P-P-D-P', 'R-P-P-R-P', 
'R-P-D-P-P', 'R-P-D-D-P', 'R-P-R-P-P', 'R-C-R-P-P', 'P-P-P-D-D', 'P-P-P-R-C', 'P-P-D-P-D', 'P-P-D-P-R', 'P-P-D-D-P', 'P-P-R-P-D', 'P-P-R-P-R', 'P-P-R-C-R', 'P-D-P-P-D', 'P-D-P-P-R', 
'P-D-P-D-P', 'P-D-P-R-P', 'P-D-D-P-P', 'P-D-D-P-R', 'P-R-P-P-D', 'P-R-P-P-R', 'P-R-P-D-P', 'P-R-P-D-D', 'P-R-P-R-P', 'P-R-P-R-C', 'P-R-C-R-P', 'D-P-D-P-R-P', 'D-P-R-P-D-P', 
'D-P-R-P-R-P', 'D-D-P-P-R-P', 'D-D-P-R-P-P', 'C-R-P-P-R-P', 'C-R-P-R-P-P', 'R-P-P-D-D-P', 'R-P-D-P-D-P', 'R-P-D-P-R-P', 'R-P-D-D-P-P', 'R-P-R-P-D-P', 'R-C-R-P-P-P', 'P-P-P-R-C-R', 
'P-P-D-D-P-R', 'P-P-R-P-D-D', 'P-P-R-P-R-C', 'P-P-R-C-R-P', 'P-D-P-D-P-R', 'P-D-P-R-P-D', 'P-D-P-R-P-R', 'P-D-D-P-P-R', 'P-D-D-P-R-P', 'P-R-P-P-D-D', 'P-R-P-P-R-C', 'P-R-P-D-P-D', 
'P-R-P-D-P-R', 'P-R-P-D-D-P', 'P-R-P-R-P-D', 'P-R-C-R-P-P', 'D-D-P-R-P-R-P', 'R-P-D-D-P-R-P', 'R-P-R-P-D-D-P', 'P-D-D-P-R-P-R', 'P-R-P-D-D-P-R', 'P-R-P-R-P-D-D']

matrix_compatibility = {
    'P-P':['PP', 'SPP'],
    'P-D':['TD'],
    'P-R':['PR'],
    'R-P':['RP'],
    'R-C':['RC'],
    'C-R':['CR'],
    'D-P':['DT'],
    'D-D':['DD']
}

heo_orders = []
bio_orders = []
for metapath in metapathss:
    matrix_orders = ['']
    for i in range(0,len(metapath)-2,2):
        x = metapath[i:i+3]
        if x == 'P-P':
            for i in range(len(matrix_orders)):
                temp = matrix_orders[i]
                matrix_orders[i] += 'PPx'
                matrix_orders.append(temp+'SPPx')
        else:
            for i in range(len(matrix_orders)):
                matrix_orders[i] += matrix_compatibility[x][0] + 'x'
    
    for i in matrix_orders:
        if len(i) != 0:
            if (i[0] == 'S' or i[0] == 'P') and i[-2] == 'P':
                heo_orders.append(i[:-1])
            else:
                bio_orders.append(i[:-1])

print(heo_orders)
print(bio_orders)

heo_orders = ['PP', 'SPP', 'PPxPP', 'SPPxPP', 'PPxSPP', 'SPPxSPP', 'PRxRP', 'PPxPRxRP', 'SPPxPRxRP', 'PRxRPxPP', 'PRxRPxSPP', 'PRxRPxPRxRP', 'PRxRCxCRxRP', 'PPxPRxRCxCRxRP', 'SPPxPRxRCxCRxRP', 'PRxRCxCRxRPxPP', 'PRxRCxCRxRPxSPP']
bio_orders = ['DT', 'RP', 'TD', 'PR', 'DTxPP', 'DTxSPP', 'DDxDT', 'CRxRP', 'RPxPP', 'RPxSPP', 'PPxTD', 'SPPxTD', 'PPxPR', 'SPPxPR', 'TDxDT', 'TDxDD', 'PRxRC', 'DTxPPxPP', 'DTxSPPxPP', 'DTxPPxSPP', 'DTxSPPxSPP', 'DTxTDxDT', 'DTxPRxRP', 'DDxDTxPP', 'DDxDTxSPP', 'CRxRPxPP', 'CRxRPxSPP', 'RPxPPxPP', 'RPxSPPxPP', 'RPxPPxSPP', 'RPxSPPxSPP', 'RPxTDxDT', 'RPxPRxRP', 'RCxCRxRP', 'PPxPPxTD', 'SPPxPPxTD', 'PPxSPPxTD', 'SPPxSPPxTD', 'PPxPPxPR', 'SPPxPPxPR', 'PPxSPPxPR', 'SPPxSPPxPR', 'PPxTDxDT', 'SPPxTDxDT', 'PPxTDxDD', 'SPPxTDxDD', 'PPxPRxRC', 'SPPxPRxRC', 'TDxDTxPP', 'TDxDTxSPP', 'TDxDTxTD', 'TDxDTxPR', 'TDxDDxDT', 'PRxRPxTD', 'PRxRPxPR', 'PRxRCxCR', 'DTxPPxTDxDT', 'DTxSPPxTDxDT', 'DTxPPxPRxRP', 'DTxSPPxPRxRP', 'DTxTDxDTxPP', 'DTxTDxDTxSPP', 'DTxPRxRPxPP', 'DTxPRxRPxSPP', 'DDxDTxPPxPP', 'DDxDTxSPPxPP', 'DDxDTxPPxSPP', 'DDxDTxSPPxSPP', 'DDxDTxPRxRP', 'CRxRPxPPxPP', 'CRxRPxSPPxPP', 'CRxRPxPPxSPP', 'CRxRPxSPPxSPP', 'CRxRPxPRxRP', 'RPxPPxTDxDT', 'RPxSPPxTDxDT', 'RPxPPxPRxRP', 'RPxSPPxPRxRP', 'RPxTDxDTxPP', 'RPxTDxDTxSPP', 'RPxTDxDDxDT', 'RPxPRxRPxPP', 'RPxPRxRPxSPP', 'RCxCRxRPxPP', 'RCxCRxRPxSPP', 'PPxPPxTDxDD', 'SPPxPPxTDxDD', 'PPxSPPxTDxDD', 
'SPPxSPPxTDxDD', 'PPxPPxPRxRC', 'SPPxPPxPRxRC', 'PPxSPPxPRxRC', 'SPPxSPPxPRxRC', 'PPxTDxDTxTD', 'SPPxTDxDTxTD', 'PPxTDxDTxPR', 'SPPxTDxDTxPR', 'PPxTDxDDxDT', 'SPPxTDxDDxDT', 'PPxPRxRPxTD', 'SPPxPRxRPxTD', 'PPxPRxRPxPR', 'SPPxPRxRPxPR', 'PPxPRxRCxCR', 'SPPxPRxRCxCR', 'TDxDTxPPxTD', 'TDxDTxSPPxTD', 'TDxDTxPPxPR', 'TDxDTxSPPxPR', 'TDxDTxTDxDT', 'TDxDTxPRxRP', 'TDxDDxDTxPP', 'TDxDDxDTxSPP', 'TDxDDxDTxPR', 'PRxRPxPPxTD', 'PRxRPxSPPxTD', 'PRxRPxPPxPR', 'PRxRPxSPPxPR', 'PRxRPxTDxDT', 'PRxRPxTDxDD', 'PRxRPxPRxRC', 'DTxTDxDTxPRxRP', 'DTxPRxRPxTDxDT', 'DTxPRxRPxPRxRP', 'DDxDTxPPxPRxRP', 'DDxDTxSPPxPRxRP', 'DDxDTxPRxRPxPP', 'DDxDTxPRxRPxSPP', 'CRxRPxPPxPRxRP', 'CRxRPxSPPxPRxRP', 'CRxRPxPRxRPxPP', 'CRxRPxPRxRPxSPP', 'RPxPPxTDxDDxDT', 'RPxSPPxTDxDDxDT', 'RPxTDxDTxTDxDT', 'RPxTDxDTxPRxRP', 'RPxTDxDDxDTxPP', 'RPxTDxDDxDTxSPP', 'RPxPRxRPxTDxDT', 'RCxCRxRPxPPxPP', 'RCxCRxRPxSPPxPP', 'RCxCRxRPxPPxSPP', 'RCxCRxRPxSPPxSPP', 'PPxPPxPRxRCxCR', 'SPPxPPxPRxRCxCR', 'PPxSPPxPRxRCxCR', 'SPPxSPPxPRxRCxCR', 'PPxTDxDDxDTxPR', 'SPPxTDxDDxDTxPR', 'PPxPRxRPxTDxDD', 'SPPxPRxRPxTDxDD', 'PPxPRxRPxPRxRC', 'SPPxPRxRPxPRxRC', 'TDxDTxTDxDTxPR', 'TDxDTxPRxRPxTD', 'TDxDTxPRxRPxPR', 'TDxDDxDTxPPxPR', 'TDxDDxDTxSPPxPR', 'TDxDDxDTxPRxRP', 'PRxRPxPPxTDxDD', 'PRxRPxSPPxTDxDD', 'PRxRPxPPxPRxRC', 'PRxRPxSPPxPRxRC', 'PRxRPxTDxDTxTD', 'PRxRPxTDxDTxPR', 'PRxRPxTDxDDxDT', 'PRxRPxPRxRPxTD', 'DDxDTxPRxRPxPRxRP', 'RPxTDxDDxDTxPRxRP', 'RPxPRxRPxTDxDDxDT', 'TDxDDxDTxPRxRPxPR', 'PRxRPxTDxDDxDTxPR', 'PRxRPxPRxRPxTDxDD']
